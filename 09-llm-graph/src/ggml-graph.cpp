#include "ggml-graph.h"

#include "ggml-ops.h" // 建节点构造函数（ggml_add / ggml_mul_mat 等，实现在 ggml-ops.cpp）

#include "ggml-kernel.h"

namespace ggml
{

    // 递归函数：把 tensor 的依赖链后序登记进 cgraph（共享子图只登记一次）。
    // visited 以 tensor 指针为 key 去重（指针唯一，不会重名误判）；叶子（op==NONE）进 leafs，算子（op 非 NONE）后序进 nodes。
    static void ggml_visit(ggml_cgraph *cgraph, ggml_tensor *tensor)
    {
        if (cgraph->visited.find(tensor) != cgraph->visited.end())
        {
            return; // 已登记过（同一节点），跳过避免重复/死循环
        }
        cgraph->visited.insert(tensor); // 登记

        // 先递归 src（后序：依赖先于消费者入列）
        for (int i = 0; i < GGML_MAX_SRC; ++i)
        {
            if (tensor->src[i])
            {
                ggml_visit(cgraph, tensor->src[i]);
            }
        }

        // 自身分类登记：叶子 vs 算子
        if (tensor->op == GGML_OP_NONE)
        {
            cgraph->leafs.push_back(tensor);
        }
        else
        {
            cgraph->nodes.push_back(tensor);
        }
    }

    void ggml_build_forward_expand(ggml_cgraph *cgraph, ggml_tensor *tensor)
    {
        cgraph->clear();
        ggml_visit(cgraph, tensor);
    }

    // ---- 执行：按 nodes 拓扑序逐个算子节点交给 ggml-kernel 分发执行 ----
    // nodes 已是「src 先于消费者」的拓扑序（见 ggml_visit），按序调用时每个节点的 src[] 数据必已就绪（或本就是叶子权重）；「op → 算子实现」的映射收敛在 ggml_kernel::compute_node，图设施不含任何算子实现。
    void ggml_graph_compute(ggml_cgraph *cgraph)
    {
        for (size_t i = 0; i < cgraph->nodes.size(); ++i)
        {
            ggml_kernel::compute_node(cgraph->nodes[i]);
        }
    }

}