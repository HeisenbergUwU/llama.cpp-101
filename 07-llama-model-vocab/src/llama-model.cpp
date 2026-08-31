// llama-model.cpp - 05 章「建 llama_model 聚合对象 + 加载权重」实现
//
// load_model 的 5 步流程（对齐上游 llama-model.cpp 的 llama_model_loader）：
//   1. llama_file 打开文件（拿 fd + size，替代裸 open/fstat）——文件只开一次
//   2. llama_mmap 映射整个文件（零拷贝，替代裸 mmap）
//   3. gguf::gguf_load 复用第 1 步的 file 解析元数据（不拷权重，只管 info，
//                      先拿到 n_tensors 供第 4 步动态算池子大小）
//   4. ggml_init  建迷你 ggml 池子（no_alloc=true：data 不占池子，留钩子；
//                      大小 = tensor 结构数 x 单结构字节数，动态算）
//   5. 循环每个 tensor：ggml_new_tensor 实例化 + ggml_set_name 定名
//                       + data = mmap.addr() + (数据段起点 + tensor.offset)  零拷贝挂载

#include "llama-model.h"

#include <cstdio>
#include <cstring>
#include <utility> // std::move

#include "gguf.h"

namespace llama
{

    // ---- 析构 ----
    // 池子里只放 ggml_tensor 结构，释放即可；mmap 是 llama_mmap 成员，
    // 会在本析构体执行完后，由它自己的析构函数自动 munmap（RAII）。
    llama_model::~llama_model()
    {
        if (ctx != nullptr)
        {
            ggml::ggml_free(ctx);
        }
    }

    // 把 gguf_type 枚举值转成可读名（对照 gguf.h 的 enum gguf_type）
    // 仅在 LLAMA_MODEL_VERBOSE 打印 KV 时用到
#if defined(LLAMA_MODEL_VERBOSE)
    static std::string kv_type_name(int64_t t)
    {
        switch (t)
        {
        case 0:
            return "UINT8";
        case 1:
            return "INT8";
        case 2:
            return "UINT16";
        case 3:
            return "INT16";
        case 4:
            return "UINT32";
        case 5:
            return "INT32";
        case 6:
            return "FLOAT32";
        case 7:
            return "BOOL";
        case 8:
            return "STRING";
        case 9:
            return "ARRAY";
        case 10:
            return "UINT64";
        case 11:
            return "INT64";
        case 12:
            return "FLOAT64";
        default:
            return "?";
        }
    }
#endif

    // ---- 加载入口：把 GGUF 文件加载成 llama_model ----
    // 用局部变量先做全部工作，最后一次性写进 llm —— 这样中途失败时
    // llm 保持「全空」的安全初始态，析构不会双重释放。
    // 注意：文件只打开一次，同一份 llama_file 既喂给 gguf_load 又给 llama_mmap。
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
        // 直接读 gguf_context.kv（只读元数据，不依赖后面的 tensor 实例化），
        // 结果存进 llm.hparams 供语义组装/前向使用（对齐上游 load_hparams 的时机）。
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

        // 打印解析出的顶级元数据：GGUF 文件头 + 数据段起点（对应测试输出的 data_offset）
        // 默认不打印（仅 05 章教学演示用）：通知多、干扰 07 的 tokenizer 测试输出。
        // 想看时用 make CXXFLAGS+='-DLLAMA_MODEL_VERBOSE' 开启。
#if defined(LLAMA_MODEL_VERBOSE)
        std::printf("gguf_context info:\n");
        std::printf("  version   = %u\n", gguf_context.version);
        std::printf("  n_tensors = %lld\n", (long long)gguf_context.n_tensors);
        std::printf("  n_kv      = %lld\n", (long long)gguf_context.n_kv);
        std::printf("  alignment = %u\n", gguf_context.alignment);
        std::printf("  offset    = 0x%llx (数据段起点)\n", (unsigned long long)gguf_context.offset);
        std::printf("  file_size = %llu\n", (unsigned long long)gguf_context.file_size);
        std::printf("  gguf_context_size = %llu\n", (unsigned long long)sizeof(gguf_context));
        for (size_t i = 0; i < gguf_context.kv.size(); i++)
        {
            const gguf::gguf_kv &kv = gguf_context.kv[i];
            std::printf("  kv[%zu] type=%lld %-12s %s = %s\n", i, (long long)kv.type,
                        kv_type_name(kv.type).c_str(), kv.key.c_str(),
                        gguf::fmt_value(kv).c_str());
        }
        // 打印所有 tensor 的元数据：名字/维度/形状/类型/段内偏移/字节数
        std::printf("  tensors (%lld):\n", (long long)gguf_context.n_tensors);
        for (size_t i = 0; i < gguf_context.info.size(); i++)
        {
            const gguf::gguf_tensor_info &ti = gguf_context.info[i];
            std::printf("    [%02zu] name=%-28s dims=%lld ne=(%lld,%lld,%lld,%lld) type=%lld offset=0x%llx nbytes=%lld\n",
                        i, ti.name.c_str(), (long long)ti.n_dims,
                        (long long)ti.ne[0], (long long)ti.ne[1],
                        (long long)ti.ne[2], (long long)ti.ne[3],
                        (long long)ti.type,
                        (unsigned long long)ti.offset,
                        (long long)ti.nbytes);
        }
#endif

        // ---- 4. ggml_init：建迷你 ggml 池子（大小按 tensor 个数动态算） ----
        // no_alloc=true：不为 tensor 数据留空间，data 留空由 mmap 填，池子只装结构。
        // 每个 tensor 占：ggml_object 头(32B) + align16(sizeof(ggml_tensor)=160B) = 192B；
        // 首个对象头从池子起点开始，故再加一份 32B；留 1KB 余量防边界溢出。
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
        // 数据段起点 = gguf_context.offset（对齐后），每个 tensor 的数据相对它再偏 ti.offset。
        // 故 data = mmap基址 + gguf_context.offset + ti.offset，直接指向文件里那块权重字节。
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
        // mapping 是局部临时，用 std::move 把映射转移给 llm.mmap
        // （llama_mmap 禁拷贝、可移动，移动后 mapping 的 addr 置空不再 munmap）。
        llm.ctx = ctx;
        // 移动而非拷贝：mmap 是独占资源只能搬家——把 mapping 的 addr/size 让给
        // llm.mmap，同时把 mapping.addr 置空；这样析构时不会对同一地址双 munmap。
        llm.mmap = std::move(mapping);
        llm.path = path;

        return true;
    }

    // ---- 06 章：模型语义组装 ----
    // 已知 tensor 名列表（plain llama，见 llama-arch.cpp 的 LLM_TN_* 表）：
    //   根  : token_embd.weight / output_norm.weight / output.weight
    //   层  : blk.{il}.attn_norm / attn_q / attn_k / attn_v / attn_output
    //                       / ffn_norm / ffn_gate / ffn_up / ffn_down （后缀统一 .weight）
    // 先建立 name -> ggml_tensor* 的查找表，再按名逐个挂接，缺一即报错。
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

        // ---- 根张量 ----
        // output.weight 是可选的（plain llama 常做「权重绑定」：lm_head 复用 token_embd，
        // 见 src/models/llama.cpp 41-46：output 缺失时直接指向 tok_embd）。故 output 允许
        // 缺失，此时 model.output 与 model.token_embd 指向同一块。
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
