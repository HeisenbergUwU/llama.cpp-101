#include "ggml-graph.h"

#include "ggml-ops.h" // 建节点构造函数（ggml_add / ggml_mul_mat 等，实现在 ggml-ops.cpp）

#include "ggml-kernel.h"

namespace ggml
{

    // 递归后序登记依赖链（共享子图只登记一次）：visited 以指针去重（指针唯一、不因重名误判）。
    // 叶子（op==NONE）进 leafs；算子后序进 nodes。
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

    // ---- 执行：按 nodes 拓扑序逐个交给 ggml-kernel 分发执行 ----
    // nodes 已拓扑序（src 先于消费者），故每节点 src[] 已就绪；「op→算子」映射在 compute_node。
    void ggml_graph_compute(ggml_cgraph *cgraph)
    {
        for (size_t i = 0; i < cgraph->nodes.size(); ++i)
        {
            ggml_kernel::compute_node(cgraph->nodes[i]);
        }
    }

}