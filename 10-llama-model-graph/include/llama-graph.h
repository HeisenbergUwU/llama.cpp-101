// llama-graph.h - 10 章「整体模型图」：namespace llama 的构建器，把具体模型前向拼成一张 ggml 图（vs 09 通用容器）。
// 用算子构造函数把 embed→12 层(attn+ffn)→output_norm→tied lm_head 登记成整图，根节点 logits。布局：激活 [dim,R]、权重 ne[0]=n_in/ne[1]=n_out、注意力逐头 2D（head_extract/concat）、无 KV cache（n_kv=N）。

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ggml-graph.h"  // ggml::ggml_cgraph
#include "llama-model.h" // namespace llama 的 Model / HParams

namespace llama
{

    // build_model_graph 的产物：一张整图 + 出 logits 的根节点。
    struct BuiltGraph
    {
        ggml::ggml_cgraph cgraph;            // 已 build（nodes=拓扑序、leafs=权重/输入）
        ggml::ggml_tensor *logits = nullptr; // [n_vocab, N]（ne={n_vocab,N}，第 t 行=token t 的 logits）
    };

    // 把完整 llama 前向建成一张整图：ctx(池子,no_alloc=false) + model(权重叶) + hp + tokens[N] + positions[N]。
    // 返回 BuiltGraph(cgraph 已 build_forward_expand(logits))。每层：①attn=RMSNorm→QKV 投影→RoPE→逐 kv 头 head_extract→score 乘→soft_max_ext(因果 mask)→mul_mat(v)→concat→wo→残差；②ffn(SwiGLU-PAR)=RMSNorm→gate/up→silu⊙up→down→残差；③out=RMSNorm→tied lm_head mul_mat→logits。
    BuiltGraph build_model_graph(ggml::ggml_context *ctx,
                                 const Model &model, const HParams &hp,
                                 const int32_t *tokens, int N, const int32_t *positions,
                                 std::string &err);

} // namespace llama
