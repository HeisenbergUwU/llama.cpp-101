// llama-model.h - 05 章「建 llama_model 聚合对象 + 加载权重」接口（最小化）。撮合 01 gguf、03
// ggml 池子、04 llama-io mmap；06 挂 Model、07 建 Vocab；11 章收推理闭环，generate() 内联在类里。

#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

#include "ggml.h"          // 03 章：namespace ggml（内存池 + 建 tensor）
#include "llama-io.h"      // 04 章：llama_mmap（文件零拷贝映射，RAII 自动 munmap）
#include "llama-hparams.h" // 06 章：HParams
#include "llama-vocab.h"   // 07 章：Vocab

namespace llama
{

    // ---- 06 章：模型语义（把平铺的裸张量挂成语义结构） ----
    // 放在 llama_model 之前，因为 llama_model 要按值持有 Model（大小需先已知）。

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

    // ---- 11 章：采样器 + 生成结果 ---- 把一个 token 的 logits 行（定长 n_vocab）压成下一个 token
    // id，默认为 greedy（argmax）；核心就一个函数，方便后续 top-k/top-p 在同一签名下换实现。
    struct Sampler
    {
        // 在 logits[0..n_vocab) 里挑 logit 最大的 token（greedy）。
        // 返回 id；n_vocab<=0 时返回 -1（调用方应保证合法）。
        int32_t sample(const float *logits, int n_vocab) const;
    };

    // generate() 的产出：完整 token 序列 + 新生成部分的统计与文本。
    struct Generation
    {
        std::vector<int32_t> token_ids; // 完整 token 序列（含输入 prompt + 新生成）
        int n_prompt = 0;               // 其中属于原 prompt 的个数（新生成从 n_prompt 起）
        int n_generated = 0;            // 实际生成的新 token 个数
        bool stopped_eog = false;       // 是否因 EOG 结束（false = 到 max_tokens 提前截断）
        std::string text;               // 新生成部分的文本（detokenize，去掉特殊 token）
    };

    // 聚合对象：已加载好权重的模型，成员生命周期都归它管。
    // 推理也挂在它身上（见 generate 成员方法）。
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

        // 06 章：语义组装结果（assemble_model 填充；供 build_model_graph 取权重）
        Model model;

        std::string path; // 模型路径（便于打印/校验）

        // ---- 11 章：自回归采样（一个文字一个文字地推理） ---- 解码循环：每步用全部历史 token
        // 重建一张前向图（无 KV cache、全量重算）、compute、对末行 greedy、边采样边打印直到 EOG/max_tokens。
        bool generate(const std::string &prompt, int max_tokens, const Sampler &sampler,
                      Generation &out, std::string &err) const;

        // 析构：先释放池子（mmap 由成员 llama_mmap 自己的析构自动 munmap）
        ~llama_model();
    };

    // 加载入口：把 GGUF 一次性加载成「完整可推理」的 llama_model（load+组装合并）：文件只打开一次，
    // 经 mmap + gguf_load 复用，每 tensor 零拷贝挂 data，最后组装进 llm.model，即可 llm.generate(...)。
    bool load_model(const std::string &path, llama_model &llm, std::string &err);

} // namespace llama
