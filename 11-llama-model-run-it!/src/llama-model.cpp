// llama-model.cpp - 05 章「建 llama_model 聚合对象 + 加载权重」实现。load_model 流程（对齐 llama_model_loader）：
// 打开一次拿 fd+size→mmap 零拷贝→gguf_load 解析（先拿 n_tensors 动态算池子）→ggml_init→逐个挂 tensor 零拷贝。

#include "llama-model.h"

#include <cstdio>
#include <cstring>
#include <utility> // std::move

#include "gguf.h"

#include "ggml-graph.h"  // ggml_init / ggml_graph_compute / ggml_free / ggml_reset（generate 用）
#include "llama-graph.h" // build_model_graph（generate 用）

namespace llama
{

    // 前向声明（定义在文件后部的语义组装；load_model 在末尾调用它，所以先声明）
    static bool assemble_model(llama_model &llm, std::string &err);

    // 估算一次前向（N 个 token）的池子字节数（generate 用）。经验公式（实测 12 层）：
    // 每层激活随 N 线性 + logits[N×n_vocab] + 静态余量，乘 2 当安全余量。
    namespace
    {
        size_t estimate_pool(const HParams &hp, int N)
        {
            const size_t logits = (size_t)N * hp.n_vocab * sizeof(float);
            const size_t layers = (size_t)hp.n_layer * (200u * 1024u + 160u * 1024u * (size_t)N);
            return (logits + layers + (4u << 20)) * 2u;
        }

        // 返回 s 中「构成完整 UTF-8 序列」的前缀字节数。中文在 SPM 词表是字节粒度
        // <0xXX> token（一字 3 字节），逐字节来要凑齐完整字符才输出，否则残缺字节->终端乱码。
        size_t complete_utf8_prefix(const std::string &s)
        {
            size_t i = 0;
            const size_t n = s.size();
            while (i < n)
            {
                const unsigned char c = (unsigned char)s[i];
                size_t len;
                if (c < 0x80)
                    len = 1; // ASCII
                else if ((c & 0xE0) == 0xC0)
                    len = 2; // U+0080~07FF
                else if ((c & 0xF0) == 0xE0)
                    len = 3; // U+0800~FFFF（中文在此）
                else if ((c & 0xF8) == 0xF0)
                    len = 4; // U+10000~10FFFF
                else
                    len = 1; // 非法字节按单字节
                if (i + len > n)
                {
                    break; // 该字符字节不足，保留为不完整尾部
                }
                i += len;
            }
            return i;
        }
    } // namespace

    // ---- 采样器：greedy（argmax 一行 logits） ----
    // 对齐上游 llama_sampler_init_greedy 的 apply（在一批候选里挑 logit 最大的 token）。
    int32_t Sampler::sample(const float *logits, int n_vocab) const
    {
        if (logits == nullptr || n_vocab <= 0)
        {
            return -1;
        }
        int32_t best = 0;
        for (int j = 1; j < n_vocab; ++j)
        {
            if (logits[j] > logits[best])
            {
                best = j;
            }
        }
        return best;
    }

    // ---- 析构 ----
    // 池子只放 ggml_tensor 结构，释放即可；mmap 是成员，析构体后由它自己的析构 RAII 自动 munmap。
    llama_model::~llama_model()
    {
        if (ctx != nullptr)
        {
            ggml::ggml_free(ctx);
        }
    }

    // ---- 加载入口：把 GGUF 文件加载成 llama_model ---- 局部变量做完全部工作最后一次性写进 llm，中途失败 llm
    // 保持「全空」安全态不致双重释放；文件只打开一次，同一份 llama_file 既喂 gguf_load 又给 llama_mmap。
    bool load_model(const std::string &path, llama_model &llm, std::string &err)
    {
        // ---- 1. 打开文件一次（llama_file：open + fstat + RAII） ----
        llama::llama_file llama_file(path.c_str());
        if (!llama_file.valid)
        {
            err = "无法打开文件: " + path;
            return false;
        }

        // ---- 2. mmap：复用第 1 步的 file，把整个文件映射进地址空间 ----
        // 零拷贝：之后每个 tensor 的 data 直接指向这块映射区。
        llama::llama_mmap mapping(llama_file);
        if (mapping.addr == nullptr)
        {
            err = "mmap 失败: " + path;
            return false;
        }

        // ---- 3. gguf::gguf_load：复用第 1 步的 file 解析元数据（不拷权重，只管 info） ----
        // 先解析以拿到 n_tensors——池子大小要按 tensor 个数动态算，所以 ggml_init 放它之后。
        gguf::gguf_context gguf_context;
        if (!gguf::gguf_load(llama_file, gguf_context, err))
        {
            return false;
        }

        // ---- 06 章：从 KV 解析超参数 ----
        // 直接读 gguf_context.kv（只读元数据），结果存进 llm.hparams 供语义组装/前向用（对齐 load_hparams）。
        if (!parse_hparams(gguf_context, llm.hparams, err))
        {
            return false;
        }

        // ---- 07 章：从 KV 构建词表 ----
        // gguf_context 此时仍有效：读 tokenizer.ggml.* 填 llm.vocab（对齐上游 load_vocab 时机）。
        if (!llm.vocab.build(gguf_context, err))
        {
            return false;
        }

        // ---- 4. ggml_init：建迷你 ggml 池子（大小按 tensor 个数动态算）---- no_alloc=true 只为结构留空间
        // （data 由 mmap 填）。每 tensor 占对象头(32B)+align16(160B)=192B，首对象头从池子起点再加 32B、留 1KB 余量。
        const size_t obj_obj_hdr = 32;                                          // sizeof(ggml_object)
        const size_t obj_tensor = ((sizeof(ggml::ggml_tensor) + 15) / 16) * 16; // 载荷对齐到 16
        const size_t pool_size = obj_obj_hdr + gguf_context.info.size() * (obj_obj_hdr + obj_tensor) + 1024;
        ggml::ggml_context *ctx = ggml::ggml_init({pool_size, nullptr, true});
        if (ctx == nullptr)
        {
            err = "ggml_init 失败（池子建不起来）";
            return false;
        }

        // ---- 5. 逐个 tensor：实例化结构+定名+零拷贝挂数据 ---- 数据段起点=gguf_context.offset（对齐后），
        // 每 tensor 相对它偏 ti.offset，故 data=mmap基址+offset+ti.offset 直接指向文件里那块权重字节。
        for (size_t i = 0; i < gguf_context.info.size(); i++)
        {
            const gguf::gguf_tensor_info &ti = gguf_context.info[i];

            ggml::ggml_tensor *t = ggml::ggml_new_tensor(
                ctx, (ggml::ggml_type)ti.type, (int)ti.n_dims, ti.ne);
            if (t == nullptr)
            {
                err = "ggml_new_tensor 失败于第 " + std::to_string(i) + " 个 tensor: " + ti.name;
                ggml::ggml_free(ctx);
                return false;
            }

            ggml::ggml_set_name(t, ti.name.c_str());
            // 零拷贝：data 直接指向 mmap 区里的权重字节（先偏移到文件里 blob 起点）
            t->data = (char *)mapping.addr + gguf_context.offset + ti.offset;

            llm.tensors.push_back(t);
        }

        // ---- 6. 语义组装：把平铺 tensor 重组进 llm.model（根 + 每层 9 权重） ----
        // llm.tensors/hparams 已就绪；组装失败则释放池子报错。对齐上游：loader 加载、model 摆成语义层。
        if (!assemble_model(llm, err))
        {
            ggml::ggml_free(ctx);
            return false;
        }

        // ---- 全部成功，一次性提交到 llm ----
        // mapping 是局部临时，用 std::move 把映射移给 llm.mmap（llama_mmap 禁拷贝可移动，移后 addr 置空防双 munmap）。
        llm.ctx = ctx;
        // 移动而非拷贝：mmap 是独占资源只能搬家——把 mapping 的 addr/size 让给 llm.mmap，
        // 同时把 mapping.addr 置空，析构时不会对同一地址双 munmap。
        llm.mmap = std::move(mapping);
        llm.path = path;

        return true;
    }

    // ---- 模型语义组装（load_model 内部；不对外）---- 已知 tensor 名（BLK 层权重见 llama-arch.cpp
    // LLM_TN_*）建 name->tensor* 查找表按名挂接、缺一报错；static 只在本文用。
    static bool assemble_model(llama_model &llm, std::string &err)
    {
        Model &model = llm.model; // 填入 llama_model 自带的语义结构成员

        // name -> tensor（名字唯一；从平铺列表线性查找即可）
        auto find = [&](const std::string &name) -> ggml::ggml_tensor *
        {
            for (ggml::ggml_tensor *t : llm.tensors)
            {
                if (std::string(t->name) == name)
                {
                    return t;
                }
            }
            return nullptr;
        };

        // ---- 根张量 ----
        // output.weight 可选（plain llama 常做权重绑定：lm_head 复用 token_embd，见 src/models/llama.cpp 41-46）。
        model.token_embd = find("token_embd.weight");
        model.output_norm = find("output_norm.weight");
        model.output = find("output.weight");
        if (model.token_embd == nullptr || model.output_norm == nullptr)
        {
            err = "缺少必需根张量（token_embd/output_norm）";
            return false;
        }
        if (model.output == nullptr)
        {
            model.output = model.token_embd; // 权重绑定：lm_head 复用 token_embd
        }

        // ---- 每层权重 bag ----
        // tinybrainbot 层号从 0 到 n_layer-1 连续、无跳号；llama 系块都是「按层打包」。
        const uint32_t n_layer = llm.hparams.n_layer;
        model.layers.assign(n_layer, Layer{});
        for (uint32_t il = 0; il < n_layer; ++il)
        {
            const std::string pre = "blk." + std::to_string(il) + ".";
            Layer &L = model.layers[il];
            L.attn_norm = find(pre + "attn_norm.weight");
            L.wq = find(pre + "attn_q.weight");
            L.wk = find(pre + "attn_k.weight");
            L.wv = find(pre + "attn_v.weight");
            L.wo = find(pre + "attn_output.weight");
            L.ffn_norm = find(pre + "ffn_norm.weight");
            L.gate = find(pre + "ffn_gate.weight");
            L.up = find(pre + "ffn_up.weight");
            L.down = find(pre + "ffn_down.weight");

            if (L.attn_norm == nullptr || L.wq == nullptr || L.wk == nullptr ||
                L.wv == nullptr || L.wo == nullptr || L.ffn_norm == nullptr ||
                L.gate == nullptr || L.up == nullptr || L.down == nullptr)
            {
                err = "第 " + std::to_string(il) + " 层缺少权重（需 attn_norm/wq/wk/wv/wo/"
                                                   "ffn_norm/gate/up/down 9 个）";
                return false;
            }
        }

        return true;
    }

    // ---- 自回归采样（llama_model 成员方法）---- 解码循环：每步用全部历史 token 重建一张前向图（无 KV cache 全量
    // 重算）、末行 greedy 边采样边打印到 EOG/max_tokens。池子只建一次、跨步 ggml_reset 复用。对齐 llama_decode+generate。
    bool llama_model::generate(const std::string &prompt, int max_tokens,
                               const Sampler &sampler, Generation &out, std::string &err) const
    {
        // err 是输出参数，先清空——循环后的「err 非空=失败」靠它判断
        err.clear();
        if (max_tokens <= 0)
        {
            err = "generate: max_tokens 必须 > 0";
            return false;
        }

        const Model &model = this->model;
        const HParams &hp = this->hparams;
        const Vocab &vocab = this->vocab;

        // ---- 初始：tokenize 输入 prompt ----
        std::vector<int32_t> toks = vocab.tokenize(prompt, false);
        if (toks.empty())
        {
            err = "generate: tokenize 未产出任何 token";
            return false;
        }
        out.token_ids = toks;
        out.n_prompt = (int)toks.size();
        out.n_generated = 0;
        out.stopped_eog = false;
        out.text.clear();

        // 逐字输出缓冲：中文等按字节 <0xXX> 分多个 token 来，攒齐一个完整 UTF-8
        // 字符才打印/计入 out.text（见 complete_utf8_prefix）；残留的不完整尾字节留在 pending。
        std::string pending;

        const int n_vocab = (int)hp.n_vocab;

        // ---- 池子只建一次、跨步复用（不每步 malloc/free）---- 大小按「最后一步最大 N=prompt+max_tokens」
        // 估一次常驻，每步 ggml_reset 归零、复用同一 buffer，最后 ggml_free 一次（对齐 ggml_gallocr）。
        const int max_N = (int)toks.size() + max_tokens;
        ggml::ggml_init_params params = {};
        params.mem_size = estimate_pool(hp, max_N);
        params.no_alloc = false;
        ggml::ggml_context *ctx = ggml::ggml_init(params);
        if (ctx == nullptr)
        {
            err = "generate: 建池失败（mem=" + std::to_string(params.mem_size) + "）";
            return false;
        }

        for (int step = 0; step < max_tokens; ++step)
        {
            const int N = (int)toks.size();

            // positions = 0..N-1（全量重算，从 0 重新标号；见 llama-graph.h 约定）
            std::vector<int32_t> pos(N);
            for (int i = 0; i < N; ++i)
            {
                pos[i] = i;
            }

            std::string berr;
            BuiltGraph bg = build_model_graph(ctx, model, hp, toks.data(), N, pos.data(), berr);
            if (bg.logits == nullptr)
            {
                err = "generate: build_model_graph 失败: " + berr;
                break;
            }

            // ② compute -> logits[N × n_vocab]
            ggml::ggml_graph_compute(&bg.cgraph);

            // ③ 采样：对末 token 行做 greedy
            const float *logits = (const float *)bg.logits->data;
            const float *last_row = logits + (size_t)(N - 1) * n_vocab;
            const int32_t next = sampler.sample(last_row, n_vocab);

            if (next < 0)
            {
                err = "generate: 采样返回非法 token";
                break;
            }

            // ④ EOG？是则结束
            if (vocab.is_eog(next))
            {
                out.stopped_eog = true;
                break;
            }

            // 否则 append + 逐字输出该 token 的文本（中文按字节 token 攒齐完整字符再打）
            toks.push_back(next);
            out.token_ids.push_back(next);
            ++out.n_generated;

            pending += vocab.detokenize(std::vector<int32_t>{next}, true);
            const size_t ready = complete_utf8_prefix(pending);
            if (ready > 0)
            {
                const std::string done = pending.substr(0, ready);
                out.text += done;
                std::fputs(done.c_str(), stdout);
                std::fflush(stdout);
                pending.erase(0, ready);
            }
        }

        // 池子跨步复用、整个循环只释放这一次
        ggml::ggml_free(ctx);

        // 循环结束分两类：正常（EOG 或跑满 max_tokens，err 为空）与失败
        // （build_model_graph / 采样出错，break 时已填 err）。失败返回 false。
        if (!err.empty())
        {
            return false;
        }
        return true;
    }

} // namespace llama
