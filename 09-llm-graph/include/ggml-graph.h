// ggml-graph.h - 迷你 ggml：计算图设施（09 章）
//
// 在 03 章加载层（ggml_context 池子 + ggml_new_tensor）之上，对齐上游 llama.cpp 的
//   ggml_mul_mat / ggml_add / ggml_build_forward_expand / ggml_graph_compute：
//   - 算子构造函数（ggml_mul_mat 等）：像上游一样「从池子 new 一个新 tensor」——
//     用 ggml_new_tensor 建出输出张量，填 op/src/op_params，返回；不真正计算。
//   - ggml_cgraph：只负责「登记」——ggml_build_forward_expand 把某个张量的依赖链
//     递归登记进 nodes[]/leafs[]（后序 + 去重），ggml_graph_compute 按拓扑序执行。
//   - 节点/叶子都是同构的 ggml_tensor（op=NONE 即叶子），没有单独的 node 类型。
//
// 权重要么是池子里带 data 的张量（本层用它做单测），要么后续指认 mmap——
// 本文件只讲「图怎么登记、怎么执行」，不关心权重从哪来。

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ggml.h" // ggml_context / ggml_tensor / ggml_new_tensor

namespace ggml
{

    // ---- 计算图（登记容器）----
    // 对应上游 ggml_cgraph：nodes[]=算子（求值顺序=拓扑序），leafs[]=叶子（常量/输入）。
    // visited 只在 build 时用来去重（共享子图只登记一次）。
    struct ggml_cgraph
    {
        std::vector<ggml_tensor *> nodes;
        std::vector<ggml_tensor *> leafs;
        std::vector<ggml_tensor *> visited; // build 去重用

        void clear()
        {
            nodes.clear();
            leafs.clear();
            visited.clear();
        }
    };

    // ---- 算子构造函数：从池子 new 一个算子节点 tensor（对齐上游 ggml_*_impl）----
    // 只登记节点、不计算；真正算在 ggml_graph_compute()。返回输出张量。
    // 权重参数是普通 ggml_tensor（叶子），会作为 src 入图、落进 leafs。
    // ne 约定（对齐上游行主序）：ne[0]=最内维（列数），ne[1]=行数。
    //   a:[R,n]（ne={n,R}）w:[n]（1D）
    ggml_tensor *ggml_rms_norm(ggml_context *ctx, ggml_tensor *a, ggml_tensor *w, float eps);
    //   a:[R,n_in]（ne={n_in,R}）b:[n_in,n_out]（ne={n_out,n_in}）-> [R,n_out]（ne={n_out,R}）
    ggml_tensor *ggml_mul_mat(ggml_context *ctx, ggml_tensor *a, ggml_tensor *b);
    //   a:[R,n] -> [R,n]；参数：slot0=n_rot(i32)，slot1=base(f32)，slot2=head_dim(i32)
    ggml_tensor *ggml_rope(ggml_context *ctx, ggml_tensor *a, int n_rot, float base, int head_dim);
    ggml_tensor *ggml_silu(ggml_context *ctx, ggml_tensor *a);
    //   a/b 同形状，逐元素乘
    ggml_tensor *ggml_mul(ggml_context *ctx, ggml_tensor *a, ggml_tensor *b);
    ggml_tensor *ggml_add(ggml_context *ctx, ggml_tensor *a, ggml_tensor *b);

    // ---- build：把 tensor 的依赖链登记成图（对应 ggml_build_forward_expand）----
    void ggml_build_forward_expand(ggml_cgraph *cgraph, ggml_tensor *tensor);

    // ---- compute：按 nodes 拓扑序逐个执行算子（对应 ggml_graph_compute）----
    void ggml_graph_compute(ggml_cgraph *cgraph);

} // namespace ggml
