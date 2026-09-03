// ggml-ops.h - 迷你 ggml：算子构造函数（建节点，09 章）。每个算子从池子 new 一个节点 tensor
// （对齐上游 ggml_*_impl），填 op/src/op_params/name；执行（按 op 分发）在 ggml-kernel，不在此。

#pragma once

#include "ggml.h" // ggml_context / ggml_tensor / ggml_new_tensor / ggml_op

namespace ggml
{

    // 算子构造函数：从池子 new 节点 tensor（对齐上游 ggml_*_impl），只登记（src 入图、落进 leafs）不计算；返回输出张量。
    //   ne 约定（对齐上游行主序）：ne[0]=最内维（列数），ne[1]=行数。如 a:[R,n]（ne={n,R}）w:[n]（1D）
    ggml_tensor *ggml_rms_norm(ggml_context *ctx, ggml_tensor *a, ggml_tensor *w, float eps);
    //   a:[n_in,R]（ne={n_in,R}）b:[n_in,n_out]（ne={n_in,n_out}）-> out:[n_out,R]（ne={n_out,R}）
    ggml_tensor *ggml_mul_mat(ggml_context *ctx, ggml_tensor *a, ggml_tensor *b);
    //   a:[n_embd,N]（每行一个 token）positions:[N]，逐行逐头（n_head×head_dim）旋转前 n_rot 维。
    //   op_params: 0=n_rot, 1=base, 2=head_dim, 3=n_head。
    ggml_tensor *ggml_rope(ggml_context *ctx, ggml_tensor *a, ggml_tensor *positions,
                           int n_rot, float base, int head_dim, int n_head);
    ggml_tensor *ggml_silu(ggml_context *ctx, ggml_tensor *a);
    //   a/b 同形状，逐元素乘
    ggml_tensor *ggml_mul(ggml_context *ctx, ggml_tensor *a, ggml_tensor *b);
    ggml_tensor *ggml_add(ggml_context *ctx, ggml_tensor *a, ggml_tensor *b);
    //   a=token_embd:[n_embd, n_vocab]  b=token ids 1D:[n_tokens] -> out:[n_embd, n_tokens]
    ggml_tensor *ggml_get_rows(ggml_context *ctx, ggml_tensor *a, ggml_tensor *b);
    //   a:[R,n] -> out:[R,n]，对每行（ne[0] 个元素）做 softmax（行求和为 1）
    ggml_tensor *ggml_softmax(ggml_context *ctx, ggml_tensor *a);
    //   a:[R,n]（ne={n,R}）-> 按 perm 重排 ne/nb（总元素数不变，不移动数据）
    ggml_tensor *ggml_permute(ggml_context *ctx, ggml_tensor *a, int p0, int p1, int p2, int p3);
    //   a（可能非行主序，如 permute 出来的 view）-> 拷成行主序连续 out。
    //   按 a->nb[] 步长逐元素取源、写进连续 dst——把「布局视图」落实成真实内存。
    ggml_tensor *ggml_cont(ggml_context *ctx, ggml_tensor *a);
    //   带掩码+缩放的 softmax（对应 ggml_soft_max_ext）：逐行做 softmax( scale·x + mask )，mask 可 NULL。
    //   a:[n_kv, n_tokens]（每行 ne[0]=n_kv 个 key 位置），mask:[n_kv, n_tokens] 或 NULL。
    ggml_tensor *ggml_soft_max_ext(ggml_context *ctx, ggml_tensor *a, ggml_tensor *mask, float scale);
    //   从「各行头连续」的激活切出第 head 个头：a:[n_embd,R]（行内 n_head×head_dim 连续）-> out:[head_dim,R]。
    //   按 a 行步长（strided）读 head 段写连续——布局切片专用小算子。
    ggml_tensor *ggml_head_extract(ggml_context *ctx, ggml_tensor *a, int head, int head_dim, int n_head);
    //   沿「特征轴（ne[0]）」拼接：a:[da, N] b:[db, N] -> out:[da+db, N]。
    //   每 token 行 = [a 的特征(da) ++ b 的特征(db)]（逐 token 交错）。供逐头结果拼回整张用。
    ggml_tensor *ggml_concat(ggml_context *ctx, ggml_tensor *a, ggml_tensor *b);

} // namespace ggml
