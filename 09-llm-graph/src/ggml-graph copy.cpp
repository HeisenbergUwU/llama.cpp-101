// ggml-graph.cpp - 迷你 ggml：计算图设施实现（09 章）
//
// 实现 ggml-graph.h：算子构造函数（用 ggml_new_tensor 从池子建算子节点 tensor）、
// ggml_build_forward_expand（递归登记）、ggml_graph_compute（按拓扑执行）。
// 数值优先复用 08 章 kernel（rms_norm/silu/rope_inplace），矩阵乘因权重就是
// 池子里 F32 张量、直接用裸 float* 实现。图只负责「拓扑 + 分发 + op_params 解读」。

#include "ggml-graph.h"

#include <cstring>

#include "kernel/kernel.h" // 8 章算子：rms_norm/silu/rope_inplace/softmax_row

namespace ggml
{

    // ---- 算子构造函数：从池子 new 一个算子节点 tensor（对齐上游 ggml_*_impl）----

    ggml_tensor *ggml_rms_norm(ggml_context *ctx, ggml_tensor *a, ggml_tensor *w, float eps)
    {
        // a:[R,n]（ne={n,R}）w:[n]（1D）-> out:[R,n]
        const int64_t ne[2] = {a->ne[0], a->ne[1]};
        ggml_tensor *out = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, ne);
        out->op = GGML_OP_RMS_NORM;
        out->src[0] = a;
        out->src[1] = w;
        ggml_set_op_params_f32(out, 0, eps);
        ggml_set_name(out, ("rms_norm(" + std::string(a->name) + ")").c_str());
        return out;
    }

    ggml_tensor *ggml_mul_mat(ggml_context *ctx, ggml_tensor *a, ggml_tensor *b)
    {
        // a:[R,n_in]（ne={n_in,R}）b:[n_in,n_out]（ne={n_out,n_in}）-> out:[R,n_out]
        const int64_t ne[2] = {b->ne[0], a->ne[1]};
        ggml_tensor *out = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, ne);
        out->op = GGML_OP_MUL_MAT;
        out->src[0] = a;
        out->src[1] = b;
        ggml_set_name(out, ("matmul(" + std::string(a->name) + ")").c_str());
        return out;
    }

    ggml_tensor *ggml_rope(ggml_context *ctx, ggml_tensor *a, int n_rot, float base, int head_dim)
    {
        // a:[R,n] -> out:[R,n]
        ggml_tensor *out = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, a->ne);
        out->op = GGML_OP_ROPE;
        out->src[0] = a;
        ggml_set_op_params_i32(out, 0, n_rot);
        ggml_set_op_params_f32(out, 1, base);
        ggml_set_op_params_i32(out, 2, head_dim);
        ggml_set_name(out, ("rope(" + std::string(a->name) + ")").c_str());
        return out;
    }

    ggml_tensor *ggml_silu(ggml_context *ctx, ggml_tensor *a)
    {
        ggml_tensor *out = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, a->ne);
        out->op = GGML_OP_SILU;
        out->src[0] = a;
        ggml_set_name(out, ("silu(" + std::string(a->name) + ")").c_str());
        return out;
    }

    ggml_tensor *ggml_mul(ggml_context *ctx, ggml_tensor *a, ggml_tensor *b)
    {
        ggml_tensor *out = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, a->ne);
        out->op = GGML_OP_MUL;
        out->src[0] = a;
        out->src[1] = b;
        ggml_set_name(out, "mul");
        return out;
    }

    ggml_tensor *ggml_add(ggml_context *ctx, ggml_tensor *a, ggml_tensor *b)
    {
        ggml_tensor *out = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, a->ne);
        out->op = GGML_OP_ADD;
        out->src[0] = a;
        out->src[1] = b;
        ggml_set_name(out, "add");
        return out;
    }

    // ---- build：递归登记依赖（对应 ggml_build_forward_expand）----
    static void ggml_visit(ggml_cgraph *cgraph, ggml_tensor *t)
    {
        for (size_t i = 0; i < cgraph->visited.size(); i++)
        {
            if (cgraph->visited[i] == t)
                return; // 去重（共享子图只登记一次）
        }
        cgraph->visited.push_back(t);
        for (int i = 0; i < GGML_MAX_SRC && t->src[i] != NULL; i++)
        {
            ggml_visit(cgraph, t->src[i]);
        }
        if (t->op == GGML_OP_NONE)
        {
            cgraph->leafs.push_back(t);
        }
        else
        {
            cgraph->nodes.push_back(t); // 拓扑序：src 先于消费者
        }
    }

    void ggml_build_forward_expand(ggml_cgraph *cgraph, ggml_tensor *tensor)
    {
        cgraph->clear();
        ggml_visit(cgraph, tensor);
    }

    // ---- compute：按拓扑序分发（对应 ggml_graph_compute）----
    void ggml_graph_compute(ggml_cgraph *cgraph)
    {
        for (size_t ni = 0; ni < cgraph->nodes.size(); ni++)
        {
            ggml_tensor *t = cgraph->nodes[ni];
            switch (t->op)
            {
            case GGML_OP_RMS_NORM:
            {
                const ggml_tensor *a = t->src[0];
                const ggml_tensor *w = t->src[1]; // 1D F32 权重
                const int64_t n = a->ne[0];
                const int64_t R = a->ne[1];
                const float eps = ggml_get_op_params_f32(t, 0);
                for (int r = 0; r < (int)R; r++)
                {
                    kernel::rms_norm((const float *)a->data + (size_t)r * n,
                                     (const float *)w->data,
                                     (int)n, eps,
                                     (float *)t->data + (size_t)r * n);
                }
                break;
            }
            case GGML_OP_MUL_MAT:
            {
                // x:[R,n_in]（ne={n_in,R}）W:[n_in,n_out]（ne={n_out,n_in}）-> out:[R,n_out]
                const ggml_tensor *x = t->src[0];
                const ggml_tensor *W = t->src[1];
                const int64_t R = x->ne[1];
                const int64_t n_in = x->ne[0];
                const int64_t n_out = W->ne[0];
                const float *xd = (const float *)x->data;
                const float *Wd = (const float *)W->data;
                float *out = (float *)t->data;
                for (int64_t r = 0; r < R; r++)
                {
                    for (int64_t j = 0; j < n_out; j++)
                    {
                        float s = 0.0f;
                        const float *xr = xd + r * n_in;
                        const float *Wj = Wd + j * n_in;
                        for (int64_t q = 0; q < n_in; q++)
                        {
                            s += xr[q] * Wj[q];
                        }
                        out[r * n_out + j] = s;
                    }
                }
                break;
            }
            case GGML_OP_ROPE:
            {
                const ggml_tensor *a = t->src[0];
                const int64_t n = a->ne[0];
                const int64_t R = a->ne[1];
                const int n_rot = ggml_get_op_params_i32(t, 0);
                const float base = ggml_get_op_params_f32(t, 1);
                const int head_dim = ggml_get_op_params_i32(t, 2);
                const int dim = head_dim > 0 ? head_dim : (int)n;
                const int n_head = dim > 0 ? (int)(n / dim) : 1;
                for (int r = 0; r < (int)R; r++)
                {
                    float *row = (float *)t->data + (size_t)r * n;
                    std::memcpy(row, (const float *)a->data + (size_t)r * n, (size_t)n * sizeof(float));
                    // 位置 = 行索引（toy 语义：无 KV cache，第 r 行即位置 r）
                    for (int h = 0; h < n_head; h++)
                    {
                        kernel::rope_inplace(row + (size_t)h * dim, n_rot, r, base);
                    }
                }
                break;
            }
            case GGML_OP_SILU:
            {
                const ggml_tensor *a = t->src[0];
                const size_t n = a->ne[0] * (size_t)a->ne[1];
                std::memcpy(t->data, a->data, n * sizeof(float));
                kernel::silu((float *)t->data, (int)n);
                break;
            }
            case GGML_OP_MUL:
            {
                const ggml_tensor *a = t->src[0];
                const ggml_tensor *b = t->src[1];
                const size_t n = a->ne[0] * (size_t)a->ne[1];
                const float *ad = (const float *)a->data;
                const float *bd = (const float *)b->data;
                float *out = (float *)t->data;
                for (size_t i = 0; i < n; i++)
                {
                    out[i] = ad[i] * bd[i];
                }
                break;
            }
            case GGML_OP_ADD:
            {
                const ggml_tensor *a = t->src[0];
                const ggml_tensor *b = t->src[1];
                const size_t n = a->ne[0] * (size_t)a->ne[1];
                const float *ad = (const float *)a->data;
                const float *bd = (const float *)b->data;
                float *out = (float *)t->data;
                for (size_t i = 0; i < n; i++)
                {
                    out[i] = ad[i] + bd[i];
                }
                break;
            }
            case GGML_OP_NONE:
                // 叶子不应出现在 nodes
                break;
            }
        }
    }

} // namespace ggml
