// ggml-graph.h - 迷你 ggml：计算图设施（09 章）
//
// 在 03 章加载层（ggml_context 池子 + ggml_new_tensor）之上，对齐上游 llama.cpp 的
//   ggml_build_forward_expand / ggml_graph_compute：
//   - ggml_cgraph：只负责「登记」——ggml_build_forward_expand 把某个张量的依赖链
//     递归登记进 nodes[]/leafs[]（后序 + 去重），ggml_graph_compute 按拓扑序执行。
//   - 节点/叶子都是同构的 ggml_tensor（op=NONE 即叶子），没有单独的 node 类型。
//   - 建节点构造函数（ggml_add / ggml_mul_mat 等）已拆到 ggml-ops.h（本头 include 它）。
//   - 「op → 具体算子」的执行分发在 ggml-kernel（compute_node），本文件不含算子实现。
//
// 权重要么是池子里带 data 的张量（本层用它做单测），要么后续指认 mmap——
// 本文件只讲「图怎么登记、怎么执行」，不关心权重从哪来。

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_set>
#include "ggml.h"     // ggml_context / ggml_tensor / ggml_new_tensor
#include "ggml-ops.h" // 建节点构造函数（ggml_add / ggml_mul_mat 等）

namespace ggml
{

    // ---- 计算图（登记容器）----
    // 对应上游 ggml_cgraph：nodes[]=算子（求值顺序=拓扑序），leafs[]=叶子（常量/输入）。
    // visited 只在 build 时用来去重（共享子图只登记一次）。
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
