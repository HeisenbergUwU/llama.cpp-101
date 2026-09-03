// ggml-kernel.h - 图 ↔ 算子的分发/组合层（09 章）
//
// 一层薄薄的桥：cg 图设施（ggml::ggml_graph_compute）只知道「一个算子节点该被
// 执行」，不知道也不该认识具体算子实现。这里把「node->op 值 → 具体算子」的
// dispatch 收敛到 namespace ggml_kernel，底层仍复用 08 章的裸 float* 单算子
// （namespace kernel，逐算子深挖的原语）。
//
// 分层：
//   ggml-graph（build/compute 骨架，按拓扑序跑节点）
//     └─ ggml-kernel（本层：compute_node 分发，拆 src/data/ne 喂给底下）
//          └─ kernel（08 章裸 float* 单算子：matmul/rms_norm/silu/rope…）
//
// compute_node 约定：
//   - 入参 node 的 op 非 NONE（叶子不进 nodes）；其 src[] 数据已就绪（拓扑序保证）。
//   - 结果写进 node->data。node->data 与 src->data 互不重叠（本课件「每节点独立
//     data」简化），in-place 类算子（silu/rope）无需担心覆盖 src。
//   - 未支持的 op 直接空转，不报错（对齐本课件最小实现）。
//
// 与 kernel 的关系：ggml_kernel 组合 kernel；kernel 不反向依赖 ggml_kernel。

#pragma once

#include "ggml.h" // ggml::ggml_tensor / ggml::ggml_op

namespace ggml_kernel
{

    // 就地执行一个算子节点：按 node->op 分发到具体 kernel 算子，结果写 node->data。
    void compute_node(ggml::ggml_tensor *node);

} // namespace ggml_kernel
