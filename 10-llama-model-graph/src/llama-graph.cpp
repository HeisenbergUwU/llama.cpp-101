// llama-graph.cpp - 10 章「整体模型图」：实现 build_model_graph，不写数值（计算全在 kernel/算子 compute），
// 只用算子构造函数把前向登记成一张拓扑图（embed→12 层 attn+ffn→output_norm→tied lm_head），根=logits。注意力逐头 2D：head_extract→mul_mat 打分→soft_max_ext(掩码+缩放)→·V→concat；无 KV cache。

#include "llama-graph.h"

#include <cmath>

#include "ggml-ops.h"

namespace llama
{

    namespace
    {

        // 建一个 1D int32 叶子（tokens / positions 用；F32 类型字节宽=4，与 int32 一致）
        ggml::ggml_tensor *make_ids(ggml::ggml_context *ctx, const int32_t *ids, int n, const char *name)
        {
            ggml::ggml_tensor *t = ggml::ggml_new_tensor_1d(ctx, ggml::GGML_TYPE_F32, n);
            for (int i = 0; i < n; ++i)
            {
                ((int32_t *)t->data)[i] = ids[i];
            }
            ggml::ggml_set_name(t, name);
            return t;
        }

        // 建因果掩码叶子 [N,N]（ne={N,N}）：s>t -> -inf（未来不可见），否则 0。
        // 与 ggml_soft_max_ext 的约定一致：逐行（每行 ne[0]=N 个 key）读，行=query。
        ggml::ggml_tensor *make_causal_mask(ggml::ggml_context *ctx, int N)
        {
            const int64_t ne[2] = {N, N};
            ggml::ggml_tensor *m = ggml::ggml_new_tensor(ctx, ggml::GGML_TYPE_F32, 2, ne);
            float *d = (float *)m->data;
            for (int t = 0; t < N; ++t)
            {
                for (int s = 0; s < N; ++s)
                {
                    d[(size_t)t * N + s] = (s > t) ? -INFINITY : 0.0f;
                }
            }
            ggml::ggml_set_name(m, "causal_mask");
            return m;
        }

    } // namespace

    BuiltGraph build_model_graph(ggml::ggml_context *ctx, const Model &model, const HParams &hp,
                                 const int32_t *tokens, int N, const int32_t *positions, std::string &err)
    {
        BuiltGraph out;

        const int n_layer = (int)hp.n_layer;
        const int n_head = (int)hp.n_head;
        const int n_head_kv = (int)hp.n_head_kv;
        const int n_gqa = (int)hp.n_gqa();
        const int n_embd_head = (int)hp.n_embd_head_k; // = 64
        const int n_rot = (int)hp.n_rot;
        const float eps = hp.f_norm_rms_eps;
        const float base = hp.rope_freq_base;
        const float scale = 1.0f / std::sqrt((float)n_embd_head);

        // ---- 输入叶子：token ids / positions / 因果掩码 ----
        ggml::ggml_tensor *tokens_t = make_ids(ctx, tokens, N, "tokens");
        ggml::ggml_tensor *pos_t = make_ids(ctx, positions, N, "positions");
        ggml::ggml_tensor *mask_t = make_causal_mask(ctx, N);

        // ---- ① embed：cur = get_rows(token_embd, tokens) -> [n_embd, N] ----
        ggml::ggml_tensor *inpL = ggml::ggml_get_rows(ctx, model.token_embd, tokens_t);
        if (!inpL)
        {
            err = "build_model_graph: 池子不足（embed get_rows）";
            return out;
        }

        // ---- ② 层循环 ----
        for (int il = 0; il < n_layer; ++il)
        {
            const Layer &L = model.layers[il];

            // —— 注意力子层 ——
            // h = RMSNorm(inpL, attn_norm)
            ggml::ggml_tensor *h = ggml::ggml_rms_norm(ctx, inpL, L.attn_norm, eps);
            // Q/K/V 投影
            ggml::ggml_tensor *Q = ggml::ggml_mul_mat(ctx, h, L.wq); // [n_head*n_embd_head, N]
            ggml::ggml_tensor *K = ggml::ggml_mul_mat(ctx, h, L.wk); // [n_head_kv*n_embd_head, N]
            ggml::ggml_tensor *V = ggml::ggml_mul_mat(ctx, h, L.wv); // 同 K
            // RoPE(Q,K)：逐 token 逐头旋转前 n_rot 维
            ggml::ggml_tensor *Qr = ggml::ggml_rope(ctx, Q, pos_t, n_rot, base, n_embd_head, n_head);
            ggml::ggml_tensor *Kr = ggml::ggml_rope(ctx, K, pos_t, n_rot, base, n_embd_head, n_head_kv);

            // 逐 kv 头注意力，每个 kv 头服务 n_gqa 个 q 头；o_head 收集后拼回 [n_embd,N]
            ggml::ggml_tensor *attn_all = NULL;
            for (int hkv = 0; hkv < n_head_kv; ++hkv)
            {
                // k/v 头切片 [n_embd_head, N]
                ggml::ggml_tensor *k_head = ggml::ggml_head_extract(ctx, Kr, hkv, n_embd_head, n_head_kv);
                ggml::ggml_tensor *v_head = ggml::ggml_head_extract(ctx, V, hkv, n_embd_head, n_head_kv);
                // ·V 需要 v 转置：[n_embd_head,N] -> [N,n_embd_head]（b 的行=输出特征）
                ggml::ggml_tensor *vT = ggml::ggml_cont(ctx, ggml::ggml_permute(ctx, v_head, 1, 0, 2, 3));

                for (int m = 0; m < n_gqa; ++m)
                {
                    const int hq = hkv * n_gqa + m; // 组内第 m 个 q 头
                    ggml::ggml_tensor *q_head = ggml::ggml_head_extract(ctx, Qr, hq, n_embd_head, n_head);
                    // score = q·k^T = mul_mat(q_head, k_head) -> [N, N]（row=query, col=key）
                    ggml::ggml_tensor *score = ggml::ggml_mul_mat(ctx, q_head, k_head);
                    // p = softmax( scale·score + causal_mask ) -> [N, N]
                    ggml::ggml_tensor *p = ggml::ggml_soft_max_ext(ctx, score, mask_t, scale);
                    // o_head = p·v = mul_mat(p, vT) -> [n_embd_head, N]
                    ggml::ggml_tensor *o_head = ggml::ggml_mul_mat(ctx, p, vT);

                    if (attn_all == NULL)
                    {
                        attn_all = o_head;
                    }
                    else
                    {
                        attn_all = ggml::ggml_concat(ctx, attn_all, o_head); // 沿特征轴拼
                    }
                }
            }
            if (!attn_all)
            {
                err = "build_model_graph: 池子不足（注意力）";
                return out;
            }
            // wo 投影 + 残差 1：inpL += wo·attn_all -> [n_embd, N]
            ggml::ggml_tensor *a = ggml::ggml_mul_mat(ctx, attn_all, L.wo);
            inpL = ggml::ggml_add(ctx, inpL, a);

            // —— 前馈子层（SwiGLU-PAR）——
            h = ggml::ggml_rms_norm(ctx, inpL, L.ffn_norm, eps);
            ggml::ggml_tensor *gate = ggml::ggml_mul_mat(ctx, h, L.gate); // [n_ff, N]
            ggml::ggml_tensor *up = ggml::ggml_mul_mat(ctx, h, L.up);     // [n_ff, N]
            ggml::ggml_tensor *g = ggml::ggml_silu(ctx, gate);            // silu(gate)
            ggml::ggml_tensor *w = ggml::ggml_mul(ctx, g, up);            // silu(gate)⊙up
            ggml::ggml_tensor *f = ggml::ggml_mul_mat(ctx, w, L.down);    // [n_embd, N]
            inpL = ggml::ggml_add(ctx, inpL, f);                          // 残差 2
        }

        // ---- ③ 输出：RMSNorm(output_norm) + tied lm_head（复用 token_embd） ----
        ggml::ggml_tensor *x = ggml::ggml_rms_norm(ctx, inpL, model.output_norm, eps);
        // logits = x·token_embd^T = mul_mat(x, token_embd) -> [n_vocab, N]
        out.logits = ggml::ggml_mul_mat(ctx, x, model.token_embd);
        if (!out.logits)
        {
            err = "build_model_graph: 池子不足（lm_head）";
            return out;
        }

        // 整图 build：从 logits 递归登记所有依赖（权重作叶子、算子按拓扑序）
        ggml::ggml_build_forward_expand(&out.cgraph, out.logits);
        return out;
    }

} // namespace llama
