// ggml-ops.h - 迷你 ggml：算子构造函数（建节点）（09 章）
// 从 ggml-graph.h 拆出的「建节点」部分：每个算子从池子 new 一个算子节点 tensor（对齐上游 ggml_*_impl），填 op/src/op_params/name 返回、不真正计算；与 ggml-graph（build/compute 骨架）职责分离，执行在 ggml-kernel。权重是带 data 的张量或后续指认 mmap，本文件只关心怎么把算子节点建出来。

#pragma once

#include "ggml.h" // ggml_context / ggml_tensor / ggml_new_tensor / ggml_op

namespace ggml
{

    // 算子构造函数：从池子 new 一个算子节点 tensor（对齐上游 ggml_*_impl），只登记节点、不计算，真正算在 ggml_graph_compute()；返回输出张量。权重参数是普通 ggml_tensor（叶子），作 src 入图、落进 leafs。
    // ne 约定（对齐上游行主序）：ne[0]=最内维（列数），ne[1]=行数；如 a:[R,n]（ne={n,R}）w:[n]（1D）
    ggml_tensor *ggml_rms_norm(ggml_context *ctx, ggml_tensor *a, ggml_tensor *w, float eps);
    //   a:[R,n_in]（ne={n_in,R}）b:[n_in,n_out]（ne={n_out,n_in}）-> [R,n_out]（ne={n_out,R}）
    ggml_tensor *ggml_mul_mat(ggml_context *ctx, ggml_tensor *a, ggml_tensor *b);
    //   a:[R,n] -> [R,n]；参数：slot0=n_rot(i32)，slot1=base(f32)，slot2=head_dim(i32)，slot3=pos(i32,默认0)
    ggml_tensor *ggml_rope(ggml_context *ctx, ggml_tensor *a, int n_rot, float base, int head_dim);
    ggml_tensor *ggml_silu(ggml_context *ctx, ggml_tensor *a);
    //   a/b 同形状，逐元素乘
    ggml_tensor *ggml_mul(ggml_context *ctx, ggml_tensor *a, ggml_tensor *b);
    ggml_tensor *ggml_add(ggml_context *ctx, ggml_tensor *a, ggml_tensor *b);

} // namespace ggml
