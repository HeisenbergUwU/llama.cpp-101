// llama-model.cpp - 05 章「建 llama_model 聚合对象 + 加载权重」实现。load_model 5 步：llama_file 打开（只开一次）
// → llama_mmap 映射（零拷贝）→ gguf_load 复用该 file（先拿 n_tensors）→ ggml_init 建池（no_alloc=true，池子大小按 tensor 数动态算而非文件大小）→ 逐 tensor new_tensor+set_name，data=mmap.addr()+offset 零拷贝挂载。

#include "llama-model.h"

#include <cstdio>
#include <cstring>
#include <utility> // std::move

#include "gguf.h"

namespace llama
{

    // ---- 析构 ----
    // 池子只放 ggml_tensor 结构，释放即可；mmap 是 llama_mmap 成员，在本析构执行完后由它自己的析构自动 munmap（RAII）。
    llama_model::~llama_model()
    {
        if (ctx != nullptr)
        {
            ggml::ggml_free(ctx);
        }
    }

    // 把 gguf_type 枚举值转成可读名（对照 gguf.h 的 enum gguf_type）
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

    // ---- 加载入口：把 GGUF 文件加载成 llama_model ----
    // 用局部变量先做全部工作、最后一次性写进 llm：中途失败时 llm 保持「全空」安全初始态、不双重释放；文件只开一次，同一份 file 既喂 gguf_load 又给 llama_mmap。
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

        // 打印解析出的顶级元数据：GGUF 文件头 + 数据段起点（对应测试输出的 data_offset）
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

        // ---- 4. ggml_init：建迷你 ggml 池子（大小按 tensor 个数动态算） ----
        // no_alloc=true：data 留空由 mmap 填，池子只装结构；每 tensor 占 ggml_object 头(32B)+align16(160B)=192B，首对象头从池子起点再加 32B，留 1KB 余量防溢出。
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
        // 数据段起点=gguf_context.offset（对齐后），每 tensor 相对它再偏 ti.offset；data=mmap基址+offset+ti.offset，直指文件里那块权重字节。
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
        // mapping 是局部临时，用 std::move 把映射转移给 llm.mmap（llama_mmap 禁拷贝、可移动；移动后 mapping.addr 置空，避免双 munmap）。
        llm.ctx = ctx;
        // 移动而非拷贝：mmap 是独占资源只能搬家——把 mapping 的 addr/size 让给
        // llm.mmap，同时把 mapping.addr 置空；这样析构时不会对同一地址双 munmap。
        llm.mmap = std::move(mapping);
        llm.path = path;

        return true;
    }

} // namespace llama
