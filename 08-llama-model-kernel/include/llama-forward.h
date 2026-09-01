// llama-forward.h - 08 章「完整前向」接口
//
// 约定（AGENTS.md）：所有代码在 namespace llama（ggml 是自己 namespace）。
// 本章目标是"完整前向"：输入一批 token（ForwardInput），算出一个
// logits[n_tokens × n_vocab]，并由 forward() 内部维护 KVSlice。
//
// 三个"接缝"（seam），是为 11 章「KV cache」预留的改刀位置，见 ROADMAP.md：
//   seam 1  KVSlice       —— 所有层历史 KV 的"不透明视图"，只暴露 n_kv
//   seam 2  build_causal_mask —— 独立的因果掩码构造（11 章改成 [n_kv×n_tokens]）
//   seam 3  forward()      —— 唯一入口，"把 K/V 存进 KVSlice"藏在内部
// 11 章只需在这三处动刀、不必重写算子；算子签名固定为裸 float*（13 章+ 深入）。

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "llama-model.h" // Model / HParams / ggml::ggml_tensor
#include "llama-hparams.h"

namespace llama
{

    // ---- 前向输入：一批 token + 各自绝对位置 + 所属序列 ----
    struct ForwardInput
    {
        std::vector<int32_t> tokens; // token id（查 token_embd 表用）
        std::vector<int32_t> pos;    // 绝对位置（RoPE 旋转用；在所属序列内累计）
        std::vector<int32_t> seq_id; // 所属序列（batch 时按它切分注意力；单序列全 0）

        // 本批有多少个 token（等价于 tokens.size()，也与 pos/seq_id 一致）
        size_t n_tokens() const { return tokens.size(); }

        static ForwardInput from_tokens(const std::vector<int32_t> &toks, int32_t seq_start)
        {
            ForwardInput in;
            in.tokens = toks;
            in.pos.resize(toks.size());
            in.seq_id.resize(toks.size(), 0);
            for (size_t i = 0; i < toks.size(); ++i)
            {
                in.pos[i] = seq_start + static_cast<int32_t>(i);
            }
            return in;
        }

        // batch 便捷构造：预分配能容纳 n 个 token 的空 batch。逐 token 用 append() 填。
        static ForwardInput make_batch(size_t n)
        {
            ForwardInput in;
            in.tokens.reserve(n);
            in.pos.reserve(n);
            in.seq_id.reserve(n);
            return in;
        }

        // batch 便捷追加：把「token + 绝对位置 + 所属序列」三元组追加进 batch。
        // 三者同时 push，维持三数组等长不变式。
        void append(int32_t token, int32_t position, int32_t seq)
        {
            tokens.push_back(token);
            pos.push_back(position);
            seq_id.push_back(seq);
        }
    };

    // ---- seam 1：KVSlice（11 章 KV cache 的落刀点） ----
    // 不透明地持有"所有层的历史 K/V"。08 章全量重算：每次 forward() 先把
    // 当前批的 N 个 token 的 K/V 全部算出来并填进去，然后 n_kv = N。
    // 注意力只读 KVSlice::n_kv，绝不假定它等于当前 token 数——这是 11 章
    // 能"只喂新增 token、把旧 K/V 留存在这"的原因。
    // 每层 K/V 布局：[kv_head][n_embd_head][kv_pos]，线性下标 = (h*n_embd_head + d)*n_kv + s。
    struct KVSlice
    {
        std::vector<std::vector<float>> k; // k[il] 长度 n_embd_head*n_head_kv*n_kv
        std::vector<std::vector<float>> v; // v[il] 同上
        int n_kv = 0;                      // 已存 K/V 的位置数

        // 按超参重设每层缓冲大小（不清 n_kv；forward 自己维护）。
        void resize(const HParams &hp)
        {
            const size_t per_layer = (size_t)hp.n_embd_head_k * hp.n_head_kv * (size_t)(n_kv > 0 ? n_kv : 1);
            k.assign(hp.n_layer, std::vector<float>(per_layer));
            v.assign(hp.n_layer, std::vector<float>(per_layer));
        }
    };

    // ---- seam 2：因果掩码 ----
    // 返回 [n_tokens × n_tokens] 的行主序矩阵：
    //   s <= t -> 0；s > t -> -inf（禁止看到未来）。
    // 11 章改成 [n_kv × n_tokens]（历史 K 行在前）。
    std::vector<float> build_causal_mask(int n_tokens);

    // ---- seam 3：forward 唯一入口 ----
    // 计算 logits[n_tokens × n_vocab]。把"每层算 K/V 并 append 进 KVSlice"藏在内部。
    // 08 章实际是"重置 KVSlice 后全量重算并填充"（kv.n_kv = n_tokens）。
    // 成功 true（logits 填好）；失败 false（err 写原因）。
    bool forward(const Model &model, const HParams &hp, const ForwardInput &input,
                 KVSlice &kv, std::vector<float> &logits, std::string &err);

} // namespace llama
