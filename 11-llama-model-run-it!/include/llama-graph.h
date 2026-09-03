// llama-graph.h - 10 章「整体模型图」：把 llama 完整前向建成一张 ggml 计算图（与 09 章图设施区分，
// 本层是具体模型前向的构建器）。布局约定：激活 [dim,R]、权重 ne[0]=n_in；注意力逐头 2D、无 KV cache。

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ggml-graph.h" // ggml::ggml_cgraph

namespace llama
{
    // 仅前向声明：build_model_graph 签名只用引用，不需完整定义；故不 include llama-model.h，
    // 避免与它（generate 内联需 include 本头）形成循环。
    struct Model;
    struct HParams;

    // build_model_graph 的产物：一张整图 + 出 logits 的根节点。
    struct BuiltGraph
    {
        ggml::ggml_cgraph cgraph;            // 已 build（nodes=拓扑序、leafs=权重/输入）
        ggml::ggml_tensor *logits = nullptr; // [n_vocab, N]（ne={n_vocab,N}，第 t 行=token t 的 logits）
    };

    // 把完整 llama 前向建成一张 ggml 计算图（embed→12 层→output_norm→tied lm_head）。
    // ctx=池子（不足则 logits==nullptr）；model/hp/tokens/positions 输入。每层流程：①注意力 RMSNorm→QKV→RoPE→逐 kv 头 head_extract→打分→soft_max_ext(掩码缩放)→·V→concat→wo→残差；②SwiGLU norm→gate/up→silu(gate)⊙up→down→残差；③输出 norm→mul_mat(token_embd)→logits。
    BuiltGraph build_model_graph(ggml::ggml_context *ctx,
                                 const Model &model, const HParams &hp,
                                 const int32_t *tokens, int N, const int32_t *positions,
                                 std::string &err);

} // namespace llama
