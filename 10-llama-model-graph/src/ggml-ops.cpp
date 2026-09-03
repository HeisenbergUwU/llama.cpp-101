// ggml-ops.cpp - 算子构造函数实现：每个算子从池子 new 节点 tensor（对齐上游 ggml_*_impl），
// 填 op/src/op_params/name 返回；不真正计算——执行在 ggml-kernel 的分发里。

#include "ggml-ops.h"

#include <string> // std::string（拼算子名）

namespace ggml
{

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
        // a:[n_in,R] b:[n_in,n_out] -> out:[n_out,R]。权重 W ne[0]=n_in、ne[1]=n_out，由 dequant_row 逐行反量化。
        // （输出每行 = 一个 token，ne[1]=R）
        const int64_t ne[2] = {b->ne[1] /*n_out*/, a->ne[1] /*R*/};
        ggml_tensor *out = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, ne);
        out->op = GGML_OP_MUL_MAT;
        out->src[0] = a;
        out->src[1] = b;
        ggml_set_name(out, ("matmul(" + std::string(a->name) + ")").c_str());
        return out;
    }

    ggml_tensor *ggml_rope(ggml_context *ctx, ggml_tensor *a, ggml_tensor *positions,
                           int n_rot, float base, int head_dim, int n_head)
    {
        // a:[n_embd, N] -> out:[n_embd, N]；positions 作 src[1]，每 token 一格。
        ggml_tensor *out = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, a->ne);
        out->op = GGML_OP_ROPE;
        out->src[0] = a;
        out->src[1] = positions;
        ggml_set_op_params_i32(out, 0, n_rot);
        ggml_set_op_params_f32(out, 1, base);
        ggml_set_op_params_i32(out, 2, head_dim);
        ggml_set_op_params_i32(out, 3, n_head);
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

    ggml_tensor *ggml_get_rows(ggml_context *ctx, ggml_tensor *a, ggml_tensor *b)
    {
        // a=token_embd:[n_embd, n_vocab]（ne={n_embd, n_vocab}，行=vocab）
        // b=token ids 1D:[n_tokens] -> out:[n_embd, n_tokens]（每 token 一行 hidden）
        const int64_t ne[2] = {a->ne[0], b->ne[0]};
        ggml_tensor *out = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, ne);
        out->op = GGML_OP_GET_ROWS;
        out->src[0] = a;
        out->src[1] = b;
        ggml_set_name(out, "get_rows");
        return out;
    }

    ggml_tensor *ggml_softmax(ggml_context *ctx, ggml_tensor *a)
    {
        ggml_tensor *out = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, a->ne);
        out->op = GGML_OP_SOFT_MAX;
        out->src[0] = a;
        ggml_set_op_params_f32(out, 0, 1.0f); // 无掩码、无缩放，scale=1
        ggml_set_name(out, "softmax");
        return out;
    }

    ggml_tensor *ggml_permute(ggml_context *ctx, ggml_tensor *a, int p0, int p1, int p2, int p3)
    {
        // 布局 op：与 a 共享同一块 data（view 机制），只重排 ne/nb 的维度顺序。
        // ne'[i] = ne[perm[i]]，nb'[i] = nb[perm[i]] —— 内存一个字节都不动。
        const int perm[4] = {p0, p1, p2, p3};
        for (int i = 0; i < 4; i++)
        {
            GGML_ASSERT(perm[i] >= 0 && perm[i] < GGML_MAX_DIMS); // 越界轴直接 abort
        }
        int64_t ne[4];
        size_t nb[4];
        for (int i = 0; i < 4; i++)
        {
            ne[i] = a->ne[perm[i]];
            nb[i] = a->nb[perm[i]];
        }
        ggml_tensor *out = ggml_view(ctx, a, 4, ne); // data 共享，view_offs=0
        for (int i = 0; i < 4; i++)
        {
            out->nb[i] = nb[i]; // 覆盖为置换后的步长
        }
        out->op = GGML_OP_PERMUTE;
        out->src[0] = a;
        ggml_set_name(out, ("permute(" + std::string(a->name) + ")").c_str());
        return out;
    }

    ggml_tensor *ggml_cont(ggml_context *ctx, ggml_tensor *a)
    {
        // 布局 op：按 a->nb[] 步长把非行主序的 view（如 permute 结果）逐元素
        // 拷回连续行主序内存。形状不变，只是「落实布局」。对齐上游 ggml_cont。
        ggml_tensor *out = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, a->ne);
        out->op = GGML_OP_CONT;
        out->src[0] = a;
        ggml_set_name(out, ("cont(" + std::string(a->name) + ")").c_str());
        return out;
    }

    ggml_tensor *ggml_soft_max_ext(ggml_context *ctx, ggml_tensor *a, ggml_tensor *mask, float scale)
    {
        // 带掩码+缩放的 softmax（对齐上游 ggml_soft_max_ext）：逐行算
        // softmax( scale·x + mask )，行内 ne[0] 个元素归一。mask 可 NULL。
        ggml_tensor *out = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, a->ne);
        out->op = GGML_OP_SOFT_MAX;
        out->src[0] = a;
        if (mask != NULL)
        {
            out->src[1] = mask;
        }
        ggml_set_op_params_f32(out, 0, scale); // slot0 = scale（默认 1.0 即普通 softmax 路径）
        ggml_set_name(out, ("soft_max_ext(" + std::string(a->name) + ")").c_str());
        return out;
    }

    ggml_tensor *ggml_head_extract(ggml_context *ctx, ggml_tensor *a, int head, int head_dim, int n_head)
    {
        // 从「每行一个 token、行内 n_head×head_dim 头连续」的激活里切第 head 个头。
        // a:[n_embd, R]（ne={n_embd,R}，n_embd=n_head*head_dim）-> out:[head_dim, R]。
        const int64_t ne_a[2] = {a->ne[0], a->ne[1]};
        const int64_t ne[2] = {(int64_t)head_dim, ne_a[1]};
        ggml_tensor *out = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, ne);
        out->op = GGML_OP_HEAD_EXTRACT;
        out->src[0] = a;
        ggml_set_op_params_i32(out, 0, head);
        ggml_set_op_params_i32(out, 1, head_dim);
        ggml_set_op_params_i32(out, 2, n_head);
        ggml_set_name(out, ("head" + std::to_string(head) + "(" + std::string(a->name) + ")").c_str());
        return out;
    }

    ggml_tensor *ggml_concat(ggml_context *ctx, ggml_tensor *a, ggml_tensor *b)
    {
        // 沿「特征轴（ne[0]）」拼接：a:[da, N] b:[db, N] -> out:[da+db, N]。
        // 每 token 行 = [a 的特征(da) ++ b 的特征(db)]（头拼接：逐 token 交错拷贝）。
        const int64_t ne[2] = {a->ne[0] + b->ne[0], a->ne[1]};
        ggml_tensor *out = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, ne);
        out->op = GGML_OP_CONCAT;
        out->src[0] = a;
        out->src[1] = b;
        ggml_set_name(out, "concat");
        return out;
    }

} // namespace ggml
