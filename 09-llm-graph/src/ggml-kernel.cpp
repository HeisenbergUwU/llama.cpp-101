// ggml-kernel.cpp - 分发/组合层实现（09 章）
// 把 ggml_graph_compute 里「按 op 调具体算子」的 switch 搬到这里：只依赖 ggml::ggml_tensor 元数据（op/src/ne/op_params/data），拆出数组喂给底层 kernel:: 裸 float* 算子，让计算图设施完全不认识算子实现。

#include "ggml-kernel.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "kernel/kernel.h"

namespace ggml_kernel
{

    // 元素总数（本层只用到 1D/2D，直接连乘四维即可）。
    static int64_t nelements(const ggml::ggml_tensor *t)
    {
        return t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3];
    }

    void compute_node(ggml::ggml_tensor *node)
    {
        const int64_t n = nelements(node);

        switch (node->op)
        {
            case ggml::GGML_OP_MUL_MAT:
            {
                // a: 输入向量（数据长 a->ne[0]）; b: 权重 [n_in, n_out]
                ggml::ggml_tensor *a = node->src[0];
                ggml::ggml_tensor *b = node->src[1];
                const int n_in  = (int)b->ne[0];
                const int n_out = (int)b->ne[1];
                // matmul 反量化权重行需要一块 scratch（长度 = 权重 n_in）
                std::vector<float> scratch((size_t)n_in > 0 ? (size_t)n_in : 1);
                kernel::matmul((const float *)a->data, b, n_in, n_out,
                               (float *)node->data, scratch.data());
                break;
            }
            case ggml::GGML_OP_ADD:
            {
                const float *x1 = (const float *)node->src[0]->data;
                const float *x2 = (const float *)node->src[1]->data;
                float *dst = (float *)node->data;
                for (int64_t j = 0; j < n; ++j)
                {
                    dst[j] = x1[j] + x2[j];
                }
                break;
            }
            case ggml::GGML_OP_MUL:
            {
                const float *x1 = (const float *)node->src[0]->data;
                const float *x2 = (const float *)node->src[1]->data;
                float *dst = (float *)node->data;
                for (int64_t j = 0; j < n; ++j)
                {
                    dst[j] = x1[j] * x2[j];
                }
                break;
            }
            case ggml::GGML_OP_RMS_NORM:
            {
                ggml::ggml_tensor *a = node->src[0]; // 输入
                ggml::ggml_tensor *w = node->src[1]; // 一维 F32 weight
                const float eps = ggml::ggml_get_op_params_f32(node, 0);
                kernel::rms_norm((const float *)a->data, (const float *)w->data,
                                 (int)a->ne[0], eps, (float *)node->data);
                break;
            }
            case ggml::GGML_OP_SILU:
            {
                // kernel::silu 是就地；先把 src 拷进 node 自己的 data 再就地算。
                const float *x = (const float *)node->src[0]->data;
                float *dst = (float *)node->data;
                std::memcpy(dst, x, (size_t)n * sizeof(float));
                kernel::silu(dst, (int)n);
                break;
            }
            case ggml::GGML_OP_ROPE:
            {
                // 旋转变换就地：先拷进 node->data 再旋转。
                // pos 存在 op_params[3]（默认 0），本课件不建模 position 张量。
                const int n_rot = ggml::ggml_get_op_params_i32(node, 0);
                const float base = ggml::ggml_get_op_params_f32(node, 1);
                const int pos = ggml::ggml_get_op_params_i32(node, 3);
                const float *x = (const float *)node->src[0]->data;
                float *dst = (float *)node->data;
                std::memcpy(dst, x, (size_t)n * sizeof(float));
                kernel::rope_inplace(dst, n_rot, pos, base);
                break;
            }
            case ggml::GGML_OP_NONE:
                // 叶子不会进 nodes（build 时按 op==NONE 分到 leafs），兜底忽略。
                break;
        }
    }

} // namespace ggml_kernel
