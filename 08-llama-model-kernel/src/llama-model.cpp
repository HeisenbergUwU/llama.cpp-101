// llama-model.cpp - 05 章「建 llama_model 聚合对象 + 加载权重」实现。load_model 5 步（对齐 llama_model_loader）：
// 1.llama_file 开一次 2.llama_mmap 映射 3.gguf_load 解析（先拿 n_tensors 算池子）4.ggml_init 建池（no_alloc data 留钩子）5.循环 tensor ggml_new_tensor+set_name，data=mmap.addr+offset+ti.offset。

#include "llama-model.h"

#include <cstdio>
#include <cstring>
#include <utility> // std::move

#include "gguf.h"

namespace llama
{

    // ---- 析构：池子只放 ggml_tensor 结构，释放即可 ----
    // mmap 是 llama_mmap 成员，本析构执行完后由它自己的析构自动 munmap（RAII）。
    llama_model::~llama_model()
    {
        if (ctx != nullptr)
        {
            ggml::ggml_free(ctx);
        }
    }

    // ---- 加载入口：把 GGUF 文件加载成 llama_model ----
    // 用局部变量做完最后一次性写进 llm，使中途失败时 llm 全空、不双重释放；文件只开一次。
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
        // 直接读 gguf_context.kv 填 llm.hparams（供语义组装/前向），对齐 load_hparams 时机。
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

        // ---- 4. ggml_init：建迷你 ggml 池子（大小按 tensor 个数动态算） ----
        // no_alloc=true 池子只装结构：每 tensor 占对象头(32B)+align16(ggml_tensor=160B)=192B，首对象头另加 32B，留 1KB 余量。
        const size_t obj_obj_hdr = 32;                                          // sizeof(ggml_object)
        const size_t obj_tensor = ((sizeof(ggml::ggml_tensor) + 15) / 16) * 16; // 载荷对齐到 16
        const size_t pool_size = obj_obj_hdr + gguf_context.info.size() * (obj_obj_hdr + obj_tensor) + 1024;
        ggml::ggml_context *ctx = ggml::ggml_init({pool_size, nullptr, true});
        if (ctx == nullptr)
        {
            err = "ggml_init 失败（池子建不起来）";
            return false;
        }

        // ---- 5. 逐个 tensor：实例化结构 + 定名 + 零拷贝挂数据 ----
        // 数据段起点=gguf_context.offset，data=mmap基址+offset+ti.offset，直接指向文件里权重字节。
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

        // ---- 全部成功，一次性提交到 llm ----
        // mapping 局部临时用 std::move 转移给 llm.mmap（禁拷贝可移动，移后 mapping.addr 置空）。
        llm.ctx = ctx;
        // 移动而非拷贝：mmap 是独占资源只能搬家——把 mapping 的 addr/size 让给
        // llm.mmap，同时把 mapping.addr 置空；这样析构时不会对同一地址双 munmap。
        llm.mmap = std::move(mapping);
        llm.path = path;

        return true;
    }

    // ---- 06 章：模型语义组装（已知 tensor 名，见 LLM_TN_* 表）----
    // 根 token_embd/output_norm/output.weight；层 blk.{il}.attn_norm/q/k/v/output/ffn_norm/gate/up/down（.weight 后缀）；先建 name->tensor 表按名挂接，缺一即报错。
    bool assemble_model(const llama_model &llm, Model &model, std::string &err)
    {
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

        // ---- 根张量：output.weight 可选（权重绑定，lm_head 复用 token_embd）----
        // 缺失时 model.output 与 token_embd 指向同一块（对齐 src/models/llama.cpp 41-46）。
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

} // namespace llama
