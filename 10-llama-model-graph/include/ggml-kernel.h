// ggml-kernel.h - 图↔算子分发/组合层（09 章）：把 node->op 分发到具体算子（namespace ggml_kernel）。
// 分层 ggml-graph(骨架)→ggml-kernel(compute_node 分发)→kernel(08 章裸 float* 单算子)。节点 op 非 NONE、src 已就绪（拓扑序），结果写 node->data；未支持 op 空转不报错。

#pragma once

#include "ggml.h" // ggml::ggml_tensor / ggml::ggml_op

namespace ggml_kernel
{

    // 就地执行一个算子节点：按 node->op 分发到具体 kernel 算子，结果写 node->data。
    void compute_node(ggml::ggml_tensor *node);

} // namespace ggml_kernel
