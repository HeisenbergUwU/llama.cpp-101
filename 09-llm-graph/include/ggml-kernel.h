// ggml-kernel.h - 图 ↔ 算子的分发/组合层（09 章）
// 薄桥：ggml-graph 只知「节点该被执行」，本层把 node->op → 具体算子 的 dispatch 收敛到 namespace ggml_kernel，底层复用 08 章裸 float* 单算子（namespace kernel）。分层：ggml-graph（骨架）→ ggml-kernel（compute_node 分发）→ kernel（单算子）。compute_node 约定：op 非 NONE（叶子不进 nodes）、src 数据已就绪（拓扑序保证）、结果写 node->data 且与 src->data 不重叠（in-place 无需担心覆盖）、不支持 op 空转不报错；ggml_kernel 组合 kernel，kernel 不反向依赖 ggml_kernel。

#pragma once

#include "ggml.h" // ggml::ggml_tensor / ggml::ggml_op

namespace ggml_kernel
{

    // 就地执行一个算子节点：按 node->op 分发到具体 kernel 算子，结果写 node->data。
    void compute_node(ggml::ggml_tensor *node);

} // namespace ggml_kernel
