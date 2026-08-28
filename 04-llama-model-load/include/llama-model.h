// llama-model.h - 04 章「建 llama_model 聚合对象 + 加载权重」接口（最小化）
//
// 只声明结构与函数，实现留 src/llama-model.cpp。04 章撮合前三章：
//   01 gguf：解析文件，给每个 tensor 的 offset/type/ne/name（临时调用）
//   03 ggml：内存池，实例化 ggml_tensor（no_alloc=true，data 留钩子）
//   mmap   ：映射整个文件，让 tensor->data 零拷贝指向文件
//
// 范围：只到「持有 110 个 ggml_tensor + mmap 映射」，不建图、不执行、
// 不做 hparams/vocab、不拷权重。故不引入上游那套 llama_model_params
// （设备 offload / kv override / 进度回调等），留待推理章再说。

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "ggml.h" // 03 章：namespace ggml（内存池 + 建 tensor，拷贝进本目录 include/）

namespace llama
{

    // 聚合对象：已加载好权重的模型，成员生命周期都归它管
    struct llama_model
    {
        // 迷你 ggml 池子句柄：所有 ggml_tensor 结构从这里分配
        ggml::ggml_context *ctx = nullptr;

        // 已建好的 tensor（指向 ctx 池内；tinybrainbot 共 110 个）
        std::vector<ggml::ggml_tensor *> tensors;

        // mmap 映射：data = mmap_addr + (数据段起点 + tensor.offset)
        void *mmap_addr = nullptr; // mmap() 基址（页对齐，含 16 对齐）
        size_t mmap_size = 0;      // 映射字节数（= 文件大小），munmap 要用
        int fd = -1;               // 打开的文件描述符，close 用

        std::string path; // 模型路径（便于打印/校验）

        // 析构：先 munmap，再 close，最后 ggml_free(ctx)（顺序不可反）
        ~llama_model();
    };

    // 加载入口：把 GGUF 文件加载成 llama_model
    // 流程：open/fstat -> mmap -> ggml_init(no_alloc) -> gguf::gguf_load
    //       -> 每个 tensor: ggml_new_tensor + set_name + data=mmap_addr+offset
    // 成功 true 填好 llm；失败 false（err 写原因）。
    bool load_model(const std::string &path, llama_model &llm, std::string &err);

} // namespace llama
