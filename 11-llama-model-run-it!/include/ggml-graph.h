// ggml-graph.h - 迷你 ggml：计算图设施（09 章）。对齐上游 ggml_build_forward_expand /
// ggml_graph_compute：ggml_cgraph 登记依赖链（后序+去重），compute 按拓扑序执行。建节点在 ggml-ops.h。

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_set>
#include "ggml.h"     // ggml_context / ggml_tensor / ggml_new_tensor
#include "ggml-ops.h" // 建节点构造函数（ggml_add / ggml_mul_mat 等）

namespace ggml
{

    // ---- 计算图（登记容器）---- 对应上游 ggml_cgraph：nodes[]=算子（拓扑序）、leafs[]=叶子；
    // visited 只在 build 时去重（共享子图只登记一次）。
    struct ggml_cgraph
    {
        std::vector<ggml_tensor *> nodes;
        std::vector<ggml_tensor *> leafs;
        std::unordered_set<ggml_tensor *> visited;
        void clear()
        {
            nodes.clear();
            leafs.clear();
            visited.clear();
        }
    };

    // ---- build：把 tensor 的依赖链登记成图（对应 ggml_build_forward_expand）----
    void ggml_build_forward_expand(ggml_cgraph *cgraph, ggml_tensor *tensor);

    // ---- compute：按 nodes 拓扑序逐个执行算子（对应 ggml_graph_compute）----
    void ggml_graph_compute(ggml_cgraph *cgraph);

} // namespace ggml
