// llama-model.cpp - 05 章「建 llama_model 聚合对象 + 加载权重」实现
//
// 依赖前几章代码：
//   01 gguf  : gguf::gguf_load 解析文件元数据（每个 tensor 的 name/ne/type/offset）
//   03 ggml  : ggml_init 建池、ggml_new_tensor 实例化结构、ggml_set_name、ggml_free
//   04 llama-io: llama_file 打开文件、llama_mmap 零拷贝映射（替代裸 open/mmap）
//
// load_model 的 5 步流程（对齐上游 llama-model.cpp 的 llama_model_loader / load_model）：
//   1. llama_file 打开文件（拿 fd + size，替代裸 open/fstat）——文件只开一次
//   2. llama_mmap 映射整个文件（零拷贝，替代裸 mmap）
//   3. ggml_init  建迷你 ggml 池子（no_alloc=true：data 不占池子，留钩子）
//   4. gguf::gguf_load 复用第 1 步的 file 解析元数据（不拷权重，只管 info）
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
        llama::llama_file llama_file(path.c_str());

        return true;
    }

} // namespace llama
