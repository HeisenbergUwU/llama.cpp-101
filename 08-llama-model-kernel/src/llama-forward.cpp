// llama-forward.cpp - 08 章「完整前向」实现
//
// 本章把「完整前向」拆成两层：
//   - kernel 层（src/kernel/cpu/kernel.cpp）：7 个无状态纯算子，可单独编译测试；
//   - 本文件：具体组件 —— KVSlice / build_causal_mask / forward，只负责把 kernel 算子
//     在层循环里串起来，不再内联任何算子实现。
//
// 方案：Scheme A（直接循环，不建计算图）。权重留在 mmap 的 data 上，由 kernel 的
// dequant_row 按行反量化（option c）进复用 scratch 缓冲、F32 累加。
//
// 数据流见 06-llama-model-assembly/model-graph.md（端到端 + 每层）。
// 三个接缝（KVSlice / build_causal_mask / forward）为 11 章 KV cache 预留。

#include "llama-forward.h"

#include <cmath>
#include <cstring>
#include <vector>

#include "kernel/kernel.h"

namespace llama
{

    // ---- seam 2：因果掩码 ----
    // [n_tokens × n_tokens] 行主序：s <= t -> 0；s > t（未来）-> -inf。
    std::vector<float> build_causal_mask(int n_tokens)
    {
        std::vector<float> mask((size_t)n_tokens * n_tokens, 0.0f);
        for (int t = 0; t < n_tokens; ++t)
        {
            for (int s = t + 1; s < n_tokens; ++s) // s > t
            {
                mask[(size_t)t * n_tokens + s] = -INFINITY;
            }
        }
        return mask;
    }

    // ---- seam 3：forward 唯一入口 ----
    // 08 章语义：全量重算 —— 重置 KVSlice 后把本批 N 个 token 一起算完填进去。
    bool forward(const Model &model, const HParams &hp, const ForwardInput &input,
                 KVSlice &kv, std::vector<float> &logits, std::string &err)
    {
        const int N = (int)input.n_tokens();
        if (N <= 0)
        {
            err = "前向输入为空（需至少 1 个 token）";
            return false;
        }
        const int n_embd = (int)hp.n_embd;
        const int n_vocab = (int)hp.n_vocab;
        const int n_layer = (int)hp.n_layer;
        const int n_head = (int)hp.n_head;
        const int n_head_kv = (int)hp.n_head_kv;
        const int n_embd_head = (int)hp.n_embd_head_k; // = n_embd_head_v = 64
        const int n_gqa = (int)hp.n_gqa();
        const int n_ff = (int)hp.n_ff;
        const int n_rot = (int)hp.n_rot;
        const float eps = hp.f_norm_rms_eps;
        const float base = hp.rope_freq_base;
        const float scale = 1.0f / std::sqrt((float)n_embd_head);

        // seam 3：把本批 K/V 存进 KVSlice。08 章每次全量重算 -> n_kv = N。
        kv.n_kv = N;
        kv.resize(hp);

        // 激活缓冲（本批 N 个 token 一起算）
        std::vector<float> cur((size_t)N * n_embd);
        std::vector<float> h((size_t)N * n_embd);      // 归一化 / 中间复用
        std::vector<float> q((size_t)N * n_head * n_embd_head);
        std::vector<float> k((size_t)N * n_head_kv * n_embd_head);
        std::vector<float> v((size_t)N * n_head_kv * n_embd_head);
        std::vector<float> attn((size_t)N * n_head * n_embd_head); // 注意力输出拼 768
        std::vector<float> gate((size_t)N * n_ff);
        std::vector<float> up((size_t)N * n_ff);
        std::vector<float> score((size_t)N);
        // 反量化缓冲：行内最长 = 2048（n_ff）取 max
        std::vector<float> scratch(n_ff > n_embd ? n_ff : n_embd);

        // mask 矩阵作为接缝独立提供；08 章正前向直接用 s>t 判断因果。
        const std::vector<float> mask = build_causal_mask(N);
        (void)mask;

        // ---- 1. embed：x[t] = token_embd_row[token[t]] ----
        for (int t = 0; t < N; ++t)
        {
            kernel::dequant_row(model.token_embd, input.tokens[t], scratch.data());
            std::memcpy(&cur[(size_t)t * n_embd], scratch.data(), (size_t)n_embd * sizeof(float));
        }

        // ---- 2. 层循环 ----
        for (int il = 0; il < n_layer; ++il)
        {
            const ggml::ggml_tensor *attn_norm = model.layers[il].attn_norm;
            const ggml::ggml_tensor *wq = model.layers[il].wq;
            const ggml::ggml_tensor *wk = model.layers[il].wk;
            const ggml::ggml_tensor *wv = model.layers[il].wv;
            const ggml::ggml_tensor *wo = model.layers[il].wo;
            const ggml::ggml_tensor *ffn_norm = model.layers[il].ffn_norm;
            const ggml::ggml_tensor *gatew = model.layers[il].gate;
            const ggml::ggml_tensor *upw = model.layers[il].up;
            const ggml::ggml_tensor *downw = model.layers[il].down;

            // ① 注意力子层
            //   RMSNorm(attn)：h[t] = RMSNorm(cur[t], attn_norm)
            for (int t = 0; t < N; ++t)
            {
                kernel::rms_norm(&cur[(size_t)t * n_embd], (const float *)attn_norm->data, n_embd, eps, &h[(size_t)t * n_embd]);
            }
            //   QKV 投影
            for (int t = 0; t < N; ++t)
            {
                const float *xt = &h[(size_t)t * n_embd];
                kernel::matmul(xt, wq, n_embd, n_embd, &q[(size_t)t * n_head * n_embd_head], scratch.data());
                kernel::matmul(xt, wk, n_embd, n_head_kv * n_embd_head, &k[(size_t)t * n_head_kv * n_embd_head], scratch.data());
                kernel::matmul(xt, wv, n_embd, n_head_kv * n_embd_head, &v[(size_t)t * n_head_kv * n_embd_head], scratch.data());
            }
            //   RoPE：q 按 n_head 个 q 头、k 按 n_head_kv 个 kv 头，各旋转前 n_rot 维；v 不转
            for (int t = 0; t < N; ++t)
            {
                const int pos = input.pos[t];
                float *qt = &q[(size_t)t * n_head * n_embd_head];
                for (int hd = 0; hd < n_head; ++hd)
                {
                    kernel::rope_inplace(qt + (size_t)hd * n_embd_head, n_rot, pos, base);
                }
                float *kt = &k[(size_t)t * n_head_kv * n_embd_head];
                for (int hd = 0; hd < n_head_kv; ++hd)
                {
                    kernel::rope_inplace(kt + (size_t)hd * n_embd_head, n_rot, pos, base);
                }
            }
            //   把本批 K/V 填进 KVSlice（这就是 11 章"append K/V"接缝的 08 版）：
            //   布局 [kv_head][d][s]，线性下标 (h*n_embd_head + d)*N + s
            for (int s = 0; s < N; ++s)
            {
                const float *ks = &k[(size_t)s * n_head_kv * n_embd_head];
                const float *vs = &v[(size_t)s * n_head_kv * n_embd_head];
                for (int hd = 0; hd < n_head_kv; ++hd)
                {
                    for (int d = 0; d < n_embd_head; ++d)
                    {
                        kv.k[il][((size_t)hd * n_embd_head + d) * N + s] = ks[(size_t)hd * n_embd_head + d];
                        kv.v[il][((size_t)hd * n_embd_head + d) * N + s] = vs[(size_t)hd * n_embd_head + d];
                    }
                }
            }
            //   Attention（GQA）：读 KVSlice（只用 kv.n_kv；s>t 掩码管因果）
            //   每个 kv 头 h 服务 q 头 3h、3h+1、3h+2（n_gqa 个）
            for (int h = 0; h < n_head_kv; ++h)
            {
                const float *kvk = &kv.k[il][(size_t)h * n_embd_head * N]; // (h,d,s)
                const float *kvv = &kv.v[il][(size_t)h * n_embd_head * N];
                for (int t = 0; t < N; ++t)
                {
                    const int q_base = t * n_head * n_embd_head + h * n_gqa * n_embd_head;
                    for (int ql = 0; ql < n_gqa; ++ql)
                    {
                        const float *qt = &q[(size_t)q_base + (size_t)ql * n_embd_head];
                        // 分值 over 所有 KV 位置 s
                        for (int s = 0; s < N; ++s)
                        {
                            if (s > t)
                            {
                                score[s] = -INFINITY;
                                continue;
                            }
                            float dot = 0.0f;
                            for (int d = 0; d < n_embd_head; ++d)
                            {
                                dot += qt[d] * kvk[(size_t)d * N + s];
                            }
                            score[s] = dot * scale;
                        }
                        kernel::softmax_row(score.data(), N);
                        // 输出 out[d] = Σ_s softmax_s × v[(d,s)]
                        float *ot = &attn[(size_t)q_base + (size_t)ql * n_embd_head];
                        for (int d = 0; d < n_embd_head; ++d)
                        {
                            float acc = 0.0f;
                            for (int s = 0; s < N; ++s)
                            {
                                acc += score[s] * kvv[(size_t)d * N + s];
                            }
                            ot[d] = acc;
                        }
                    }
                }
            }
            //   wo：a[t] = matmul(attn[t], wo)；残差 1：cur[t] += a[t]
            for (int t = 0; t < N; ++t)
            {
                kernel::matmul(&attn[(size_t)t * n_head * n_embd_head], wo, n_embd, n_embd, &h[(size_t)t * n_embd], scratch.data());
                float *ct = &cur[(size_t)t * n_embd];
                const float *at = &h[(size_t)t * n_embd];
                for (int j = 0; j < n_embd; ++j)
                {
                    ct[j] += at[j];
                }
            }

            // ② 前馈子层（SwiGLU-PAR）
            //   RMSNorm(ffn)：h[t] = RMSNorm(cur[t], ffn_norm)
            for (int t = 0; t < N; ++t)
            {
                kernel::rms_norm(&cur[(size_t)t * n_embd], (const float *)ffn_norm->data, n_embd, eps, &h[(size_t)t * n_embd]);
            }
            //   gate / up 各看原始输入 h 再乘（PAR）
            for (int t = 0; t < N; ++t)
            {
                kernel::matmul(&h[(size_t)t * n_embd], gatew, n_embd, n_ff, &gate[(size_t)t * n_ff], scratch.data());
                kernel::matmul(&h[(size_t)t * n_embd], upw, n_embd, n_ff, &up[(size_t)t * n_ff], scratch.data());
            }
            //   w = silu(gate) ⊙ up（就地：gate 缓冲覆盖）
            for (int t = 0; t < N; ++t)
            {
                float *gt = &gate[(size_t)t * n_ff];
                const float *ut = &up[(size_t)t * n_ff];
                kernel::silu(gt, n_ff);
                for (int j = 0; j < n_ff; ++j)
                {
                    gt[j] *= ut[j];
                }
            }
            //   down：f[t] = matmul(w, down)；残差 2：cur[t] += f[t]
            for (int t = 0; t < N; ++t)
            {
                kernel::matmul(&gate[(size_t)t * n_ff], downw, n_ff, n_embd, &h[(size_t)t * n_embd], scratch.data());
                float *ct = &cur[(size_t)t * n_embd];
                const float *ft = &h[(size_t)t * n_embd];
                for (int j = 0; j < n_embd; ++j)
                {
                    ct[j] += ft[j];
                }
            }
        }

        // ---- 3. 输出归一 + lm_head（tied：复用 token_embd） ----
        for (int t = 0; t < N; ++t)
        {
            kernel::rms_norm(&cur[(size_t)t * n_embd], (const float *)model.output_norm->data, n_embd, eps, &h[(size_t)t * n_embd]);
        }
        logits.resize((size_t)N * n_vocab);
        for (int t = 0; t < N; ++t)
        {
            const float *ht = &h[(size_t)t * n_embd];
            float *lt = &logits[(size_t)t * n_vocab];
            for (int j = 0; j < n_vocab; ++j)
            {
                kernel::dequant_row(model.token_embd, j, scratch.data()); // tied：同 token_embd
                float acc = 0.0f;
                for (int i = 0; i < n_embd; ++i)
                {
                    acc += ht[i] * scratch[i];
                }
                lt[j] = acc;
            }
        }

        return true;
    }

} // namespace llama
