// ggml-ops.cpp - 算子构造函数实现（09 章）
// 每个算子从池子 new 一个算子节点 tensor（对齐上游 ggml_*_impl），填 op/src/op_params/name 返回输出张量；不真正计算，执行在 ggml-kernel 的分发里。

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
        ggml_set_op_params_i32(out, 3, 0); // slot3=pos：本课件不建模 position 张量，默认 0
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

} // namespace ggml
