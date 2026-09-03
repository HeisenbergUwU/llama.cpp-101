// ggml-kernel.h - 图 ↔ 算子的分发/组合层（09 章）：把「node->op → 具体算子」的 dispatch 收敛到
// namespace ggml_kernel，底层复用 08 章裸 float*。compute_node：op 非 NONE、src.data 已就绪（拓扑序），把结果写 node->data；未支持 op 空转。

#pragma once

#include "ggml.h" // ggml::ggml_tensor / ggml::ggml_op

namespace ggml_kernel
{

    // 就地执行一个算子节点：按 node->op 分发到具体 kernel 算子，结果写 node->data。
    void compute_node(ggml::ggml_tensor *node);

} // namespace ggml_kernel
