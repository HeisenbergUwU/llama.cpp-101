// llama-model.cpp - 05 章「建 llama_model 聚合对象 + 加载权重」实现
//
// 依赖前几章代码：
//   01 gguf  : gguf::gguf_load 解析文件元数据（每个 tensor 的 name/ne/type/offset）
//   03 ggml  : ggml_init 建池、ggml_new_tensor 实例化结构、ggml_set_name、ggml_free
//   04 llama-io: llama_file 打开文件、llama_mmap 零拷贝映射（替代裸 open/mmap）
//
// load_model 的 5 步流程（对齐上游 llama-model.cpp 的 llama_model_loader / load_model）：
//   1. llama_file 打开文件（拿 fd + size，替代裸 open/fstat）
//   2. llama_mmap 映射整个文件（零拷贝，替代裸 mmap）
//   3. ggml_init  建迷你 ggml 池子（no_alloc=true：data 不占池子，留钩子）
//   4. gguf::gguf_load 解析元数据（不拷权重，只管 info）
//   5. 循环每个 tensor：ggml_new_tensor 实例化 + ggml_set_name 定名
//                       + data = mmap.addr() + (数据段起点 + tensor.offset)  零拷贝挂载

#include "llama-model.h"

#include <cstdio>
#include <cstring>
#include <utility> // std::move

#include "gguf.h" // 01 章解析器（拷贝进本目录 include/）

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

    // ---- 加载入口：把 GGUF 文件加载成 llama_model ----
    // 用局部变量先做全部工作，最后一次性写进 llm —— 这样中途失败时
    // llm 保持「全空」的安全初始态，析构不会双重释放。
    bool load_model(const std::string &path, llama_model &llm, std::string &err)
    {
        // ---- 1. 打开文件（llama_file：open + fstat + RAII） ----
        llama::llama_file file(path.c_str());
        if (!file.valid)
        {
            err = "无法打开文件: " + path;
            return false;
        }

        // ---- 2. mmap：把整个文件映射进地址空间（llama_mmap：零拷贝） ----
        llama::llama_mmap mapping(file);
        if (mapping.addr == nullptr)
        {
            err = "mmap 失败: " + path;
            return false;
        }

        // ---- 3. ggml_init：建迷你 ggml 池子 ----
        // no_alloc=true：不为 tensor 数据在池子里留空间，data 留空由 mmap 填。
        // mem_size：池子只放「110 个 ggml_tensor 结构 + 对象头」，够用即可。
        const size_t pool_size = 4 * 1024 * 1024; // 4MB，装 110 个 tensor 结构绰绰有余
        ggml::ggml_context *ctx = ggml::ggml_init({pool_size, nullptr, true});
        if (ctx == nullptr)
        {
            err = "ggml_init 失败（池子建不起来）";
            return false;
        }

        // ---- 4. gguf::gguf_load：解析元数据（不拷权重） ----
        gguf::gguf_context info;
        if (!gguf::gguf_load(path, info, err))
        {
            ggml::ggml_free(ctx);
            return false;
        }

        // ---- 5. 逐个 tensor：实例化结构 + 定名 + 零拷贝挂数据 ----
        // 数据段起点 = info.offset（对齐后），每个 tensor 的数据相对它再偏 info[i].offset。
        // 故 data = mmap基址 + info.offset + ti.offset，直接指向文件里那块权重。
        for (size_t i = 0; i < info.info.size(); i++)
        {
            const gguf::gguf_tensor_info &ti = info.info[i];

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
            t->data = (char *)mapping.addr + info.offset + ti.offset;

            llm.tensors.push_back(t);
        }

        // ---- 全部成功，一次性提交到 llm ----
        // mmap 拷贝语义：mapping 是局部临时，用 std::move 把映射转移给 llm.mmap
        // （llama_mmap 禁拷贝、可移动，移动后 mapping 的 addr 置空不再 munmap）。
        llm.ctx = ctx;
        llm.mmap = std::move(mapping);
        llm.path = path;

        return true;
    }

} // namespace llama
