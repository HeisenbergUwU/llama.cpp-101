// ggml-kernel.cpp - 分发/组合层实现（09 章）。把 ggml_graph_compute 里「按 op 调具体算子」的 switch
// 搬到这：只依赖 ggml_tensor 元数据（op/src/ne/op_params/data），拆数组喂给底层 kernel 裸 float* 算子。

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
            // a:[R,n_in]（ne={n_in,R}）逐行 × b:[n_in,n_out] -> out:[R,n_out]。按 a 行循环，每行 r
            // 喂 kernel::matmul 写 out 第 r 行（b 每行由 dequant_row 反量化，F32 即 memcpy）。
            ggml::ggml_tensor *a = node->src[0];
            ggml::ggml_tensor *b = node->src[1];
            const int n_in = (int)b->ne[0];
            const int n_out = (int)b->ne[1];
            const int R = (int)a->ne[1]; // 批行数（激活的 token 数）
            // matmul 反量化权重行需要一块 scratch（长度 = 权重 n_in）
            std::vector<float> scratch((size_t)n_in > 0 ? (size_t)n_in : 1);
            const float *ad = (const float *)a->data;
            float *od = (float *)node->data;
            for (int r = 0; r < R; ++r)
            {
                kernel::matmul(ad + (size_t)r * n_in, b, n_in, n_out,
                               od + (size_t)r * n_out, scratch.data());
            }
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
            // a:[n_embd, R]（每行一个 token、行内 n_embd 个元素）w:[n_embd]（1D F32）
            // -> 同形状。kernel::rms_norm 每次处理一行，按行（token）循环。
            ggml::ggml_tensor *a = node->src[0]; // 输入
            ggml::ggml_tensor *w = node->src[1]; // 一维 F32 weight
            const float eps = ggml::ggml_get_op_params_f32(node, 0);
            const int n_embd = (int)a->ne[0];
            const int R = (int)a->ne[1];
            const float *ad = (const float *)a->data;
            float *od = (float *)node->data;
            for (int r = 0; r < R; ++r)
            {
                kernel::rms_norm(ad + (size_t)r * n_embd, (const float *)w->data,
                                 n_embd, eps, od + (size_t)r * n_embd);
            }
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
            // 旋转：src[0]=激活 [n_embd,N]（每行一 token、行内各头连续），src[1]=positions[N]（int32）；
            // 逐行逐头（n_head 个、每头 head_dim 维）旋转前 n_rot 维。op_params: 0=n_rot,1=base,2=head_dim,3=n_head。
            const int n_rot = ggml::ggml_get_op_params_i32(node, 0);
            const float base = ggml::ggml_get_op_params_f32(node, 1);
            const int head_dim = ggml::ggml_get_op_params_i32(node, 2);
            const int n_head = ggml::ggml_get_op_params_i32(node, 3);
            const float *x = (const float *)node->src[0]->data;
            const int32_t *pos = (const int32_t *)node->src[1]->data;
            float *dst = (float *)node->data;
            const int64_t R = node->src[0]->ne[1];       // token 数（行）
            const int64_t row_len = node->src[0]->ne[0]; // n_embd
            for (int64_t r = 0; r < R; ++r)
            {
                std::memcpy(dst + r * row_len, x + r * row_len, (size_t)row_len * sizeof(float));
                for (int h = 0; h < n_head; ++h)
                {
                    kernel::rope_inplace(dst + r * row_len + (int64_t)h * head_dim,
                                         n_rot, pos[r], base);
                }
            }
            break;
        }
        case ggml::GGML_OP_NONE:
            // 叶子不会进 nodes（build 时按 op==NONE 分到 leafs），兜底忽略。
            break;
        case ggml::GGML_OP_GET_ROWS:
        {
            // a=token_embd:[n_embd, n_vocab]（可能是 F16！），b=token ids 1D:[n_tokens]，输出
            // [n_embd, n_tokens]：逐 token 把 a 对应行反量化进输出——不能用 memcpy，须走 dequant_row。
            ggml::ggml_tensor *a = node->src[0];
            ggml::ggml_tensor *b = node->src[1];
            const int64_t n_embd = a->ne[0];
            const int64_t n_tokens = b->ne[0];
            const int32_t *ids = (const int32_t *)b->data;
            float *dst = (float *)node->data;
            for (int64_t t = 0; t < n_tokens; t++)
            {
                const int64_t row = ids[t];
                kernel::dequant_row(a, (int)row, dst + t * n_embd);
            }
            break;
        }
        case ggml::GGML_OP_SOFT_MAX:
        {
            // 带掩码+缩放 softmax：逐行（ne[0] 个元素）先套 scale·x+mask 再归一。
            // mask 在 src[1]（可能为 NULL）；scale 在 op_params[0]（默认 1.0）。
            const float scale = ggml::ggml_get_op_params_f32(node, 0);
            const float *x = (const float *)node->src[0]->data;
            const float *mask = node->src[1] ? (const float *)node->src[1]->data : NULL;
            float *dst = (float *)node->data;
            const int64_t row_len = node->ne[0];
            const int64_t n_rows = node->ne[1];
            for (int64_t r = 0; r < n_rows; r++)
            {
                float *row = dst + r * row_len;
                std::memcpy(row, x + r * row_len, (size_t)row_len * sizeof(float));
                kernel::soft_max_ext(row, mask ? mask + r * row_len : NULL, scale, (int)row_len);
            }
            break;
        }
        case ggml::GGML_OP_PERMUTE:
            // 布局 op：data 已在建节点时通过 view 共享到源，ne/nb 也已重排，无需计算。
            break;
        case ggml::GGML_OP_CONT:
        {
            // 布局 op：按 src 的 nb[] 步长把非行主序视图逐元素拷进连续 dst。
            // 源偏移 = Σ_d idx[d]·src->nb[d]；目标连续 = Σ_d idx[d]·dst->nb[d]。
            const ggml::ggml_tensor *a = node->src[0];
            const char *src = (const char *)a->data;
            char *dst = (char *)node->data;
            const int64_t ne0 = a->ne[0], ne1 = a->ne[1], ne2 = a->ne[2], ne3 = a->ne[3];
            for (int64_t i3 = 0; i3 < ne3; i3++)
                for (int64_t i2 = 0; i2 < ne2; i2++)
                    for (int64_t i1 = 0; i1 < ne1; i1++)
                        for (int64_t i0 = 0; i0 < ne0; i0++)
                        {
                            const size_t s = (size_t)i0 * a->nb[0] +
                                             (size_t)i1 * a->nb[1] +
                                             (size_t)i2 * a->nb[2] +
                                             (size_t)i3 * a->nb[3];
                            const size_t d = (size_t)((i0) +
                                                      i1 * ne0 +
                                                      i2 * ne0 * ne1 +
                                                      i3 * ne0 * ne1 * ne2) *
                                             sizeof(float);
                            std::memcpy(dst + d, src + s, a->nb[0]);
                        }
            break;
        }
        case ggml::GGML_OP_HEAD_EXTRACT:
        {
            // 从「每行一个 token、行内 n_head×head_dim 头连续」的激活切第 head 个头。
            // a:[n_embd, R]（行步长 n_embd）-> out:[head_dim, R]（每行连续 head_dim）。
            const ggml::ggml_tensor *a = node->src[0];
            const int head = ggml::ggml_get_op_params_i32(node, 0);
            const int head_dim = ggml::ggml_get_op_params_i32(node, 1);
            const int64_t R = a->ne[1];
            const int64_t n_embd = a->ne[0];
            const float *src = (const float *)a->data;
            float *dst = (float *)node->data;
            for (int64_t r = 0; r < R; ++r)
            {
                std::memcpy(dst + r * head_dim,
                            src + r * n_embd + (int64_t)head * head_dim,
                            (size_t)head_dim * sizeof(float));
            }
            break;
        }
        case ggml::GGML_OP_CONCAT:
        {
            // 沿特征轴（ne[0]）拼接：a:[da,N] b:[db,N] -> out:[da+db,N]。
            // 逐 token 交错拷贝：out[t] = [a[t](da) ++ b[t](db)]。
            const ggml::ggml_tensor *a = node->src[0];
            const ggml::ggml_tensor *b = node->src[1];
            const int64_t da = a->ne[0], db = b->ne[0];
            const int64_t N = a->ne[1];
            const float *ad = (const float *)a->data;
            const float *bd = (const float *)b->data;
            float *dst = (float *)node->data;
            for (int64_t t = 0; t < N; ++t)
            {
                std::memcpy(dst + t * (da + db), ad + t * da, (size_t)da * sizeof(float));
                std::memcpy(dst + t * (da + db) + da, bd + t * db, (size_t)db * sizeof(float));
            }
            break;
        }
        }
    }

} // namespace ggml_kernel
