// llama-model.h - 05 章「建 llama_model 聚合对象 + 加载权重」接口（最小化）
//
// 只声明结构与函数，实现留 src/llama-model.cpp。05 章撮合前几章：
//   01 gguf：解析文件，给每个 tensor 的 offset/type/ne/name
//   03 ggml：内存池，实例化 ggml_tensor（no_alloc=true，data 留钩子）
//   04 llama-io：llama_mmap 映射整个文件，让 tensor->data 零拷贝指向文件
//
// 范围：只到「持有 110 个 ggml_tensor + mmap 映射」，不建图、不执行、
// 不做 hparams/vocab、不拷权重。故不引入上游的 llama_model_params
// （设备 offload / kv override / 进度回调等），留待推理章再说。

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "ggml.h"    // 03 章：namespace ggml（内存池 + 建 tensor）
#include "llama-io.h" // 04 章：llama_mmap（文件零拷贝映射，RAII 自动 munmap）

namespace llama
{

    // 聚合对象：已加载好权重的模型，成员生命周期都归它管
    struct llama_model
    {
        // 文件映射：整个模型文件零拷贝挂进地址空间（RAII：析构自动 munmap）。
        // 注意：它在 llama_mmap 里禁拷贝、可移动，所以 llama_model 默认构造用空映射。
        llama::llama_mmap mmap;

        // 迷你 ggml 池子句柄：所有 ggml_tensor 结构从这里分配
        ggml::ggml_context *ctx = nullptr;

        // 已建好的 tensor（指向 ctx 池内；tinybrainbot 共 110 个）
        std::vector<ggml::ggml_tensor *> tensors;

        std::string path; // 模型路径（便于打印/校验）

        // 析构：先释放池子（mmap 由成员 llama_mmap 自己的析构自动 munmap）
        ~llama_model();
    };

    // 加载入口：把 GGUF 文件加载成 llama_model
    // 流程：llama_file 打开一次 -> llama_mmap 映射（同一 file）
    //       -> gguf::gguf_load 复用该 file -> 每个 tensor: ggml_new_tensor + set_name
    //                            + data = mmap.addr() + info.offset + ti.offset
    // 成功 true 填好 llm；失败 false（err 写原因）。
    bool load_model(const std::string &path, llama_model &llm, std::string &err);

} // namespace llama
