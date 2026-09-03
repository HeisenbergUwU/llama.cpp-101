// llama-model.h - 05 章「建 llama_model + 加载权重」接口（最小化）；只声明结构与函数，实现留 src/llama-model.cpp。
// 05 章撮合 01 gguf(offset/type/ne/name) + 03 ggml(池子实例化,no_alloc=true) + 04 llama-io(mmap 零拷贝)；范围只到「持有 110 个 tensor + mmap」，不建图/不执行/不做 hparams/vocab，故不引入上游 llama_model_params（offload/kv override 等）。

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "ggml.h"          // 03 章：namespace ggml（内存池 + 建 tensor）
#include "llama-io.h"      // 04 章：llama_mmap（文件零拷贝映射，RAII 自动 munmap）
#include "llama-hparams.h" // 06 章：HParams（assemble_model 需要 n_layer 校验层数）
#include "llama-vocab.h"   // 07 章：Vocab（load_model 从 KV 建词表）

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

        // 06 章：解析出的超参数（load_model 内部从 KV 读，供语义组装 / 前向用）
        HParams hparams;

        // 07 章：词表（load_model 内部从 tokenizer.ggml.* KV 建，供 tokenize/detokenize）
        Vocab vocab;

        std::string path; // 模型路径（便于打印/校验）

        // 析构：先释放池子（mmap 由成员 llama_mmap 自己的析构自动 munmap）
        ~llama_model();
    };

    // 加载入口：把 GGUF 文件加载成 llama_model。
    // 流程：llama_file 开一次 -> llama_mmap 映射 -> gguf_load 复用该 file -> 每 tensor ggml_new_tensor+set_name+data=mmap.addr()+offset+ti.offset。成功 true 填好 llm，失败 false（err 写原因）。
    bool load_model(const std::string &path, llama_model &llm, std::string &err);

    // ---- 06 章：模型语义（把平铺的裸张量挂成语义结构） ----

    // 单个 transformer 层的权重 bag（字段名对齐上游 llama-model.h 的 llama_layer）。
    // plain llama 每层 9 个权重，全部指向 llm.tensors 里对应的 ggml_tensor。
    struct Layer
    {
        ggml::ggml_tensor *attn_norm = nullptr; // RMSNorm（残差前）
        ggml::ggml_tensor *wq = nullptr;        // Q 投影（blk.%d.attn_q）
        ggml::ggml_tensor *wk = nullptr;        // K 投影（blk.%d.attn_k）
        ggml::ggml_tensor *wv = nullptr;        // V 投影（blk.%d.attn_v）
        ggml::ggml_tensor *wo = nullptr;        // 注意力输出投影（blk.%d.attn_output）
        ggml::ggml_tensor *ffn_norm = nullptr;  // RMSNorm（FFN 前）
        ggml::ggml_tensor *gate = nullptr;      // GLU 的 gate（blk.%d.ffn_gate）
        ggml::ggml_tensor *up = nullptr;        // GLU 的 up（blk.%d.ffn_up）
        ggml::ggml_tensor *down = nullptr;      // 投影回 n_embd（blk.%d.ffn_down）
    };

    // 有语义的模型：根张量 + 每层一个 bag。
    // 只存指向 llm.tensors 的指针，不复制数据、不拥有内存（生命周期仍归 llama_model）。
    struct Model
    {
        ggml::ggml_tensor *token_embd = nullptr;  // token_embd.weight：token -> hidden
        ggml::ggml_tensor *output_norm = nullptr; // output_norm.weight：最终 RMSNorm
        ggml::ggml_tensor *output = nullptr;      // output.weight：hidden -> logits；缺失时绑定 token_embd
        std::vector<Layer> layers;                // n_layer 个，每个 9 个权重
    };

    // 按 tensor 名把 llm.tensors 平铺列表重组进 model 语义结构；期望层数取 llm.hparams.n_layer（校验 layers.size()==n_layer）。
    // 根张量（token_embd/output_norm/output）或任一层缺权重即报错；成功 true 填好 model，失败 false（err 写原因）。
    bool assemble_model(const llama_model &llm, Model &model, std::string &err);

} // namespace llama
