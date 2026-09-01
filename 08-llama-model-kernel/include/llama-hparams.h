// llama-hparams.h - 06 章「模型语义：超参数 HParams」接口
//
// 从 GGUF 的 KV 元数据读出「决定架构的超参数」，并算出派生维度。
// 06 章只做 tinybrainbot 这种 plain llama 需要的最小字段集——不抄上游动辄
// 上百个字段的 llama_hparams（llama.cpp/src/llama-hparams.h），只保留能
// 把本模型看懂的 ~10 个 + 派生 getter。
//
// 与上游对应关系（详见 reference.md）：
//   KV 读取   : llama.cpp/src/llama-model.cpp 的 llama_model_base::load_hparams
//   派生 getter: llama.cpp/src/llama-hparams.cpp 的 n_head/n_embd_head_k/n_gqa/
//               n_embd_k_gqa/n_rot
//   KV 键名   : llama.cpp/src/llama-arch.cpp 的 LLM_KV_* 字符串表（"<arch>.<subkey>"）

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gguf
{
    struct gguf_context; // 前向声明：只依赖它的 kv 列表做解析，不必 include 全量头
}

namespace llama
{

    // 一个普通 transformer 的最小超参数集。
    // 直接值来自 GGUF KV；派生 getter 从上游算法算出（见成员后注释）。
    struct HParams
    {
        // ---- 直接读 KV ----
        uint32_t n_embd = 0;             // 隐藏维度（llama.embedding_length）        实测 768
        uint32_t n_layer = 0;            // transformer 层数（llama.block_count）     实测 12
        uint32_t n_head = 0;             // query 头数（llama.attention.head_count）    实测 12
        uint32_t n_head_kv = 0;          // kv 头数（llama.attention.head_count_kv）    实测 4
        uint32_t n_ff = 0;               // 前馈中间维度（llama.feed_forward_length）   实测 2048
        uint32_t n_embd_head_k = 0;      // 每头 k 维度（llama.attention.key_length；默认 n_embd/n_head） 实测 64
        uint32_t n_embd_head_v = 0;      // 每头 v 维度（llama.attention.value_length；默认 n_embd/n_head） 实测 64
        uint32_t n_rot = 0;              // RoPE 旋转维度（llama.rope.dimension_count；默认 n_embd_head_k） 实测 64
        uint32_t n_vocab = 0;            // 词表大小（llama.vocab_size）               实测 32000
        float rope_freq_base = 10000.0f; // 频率基（llama.rope.freq_base；默认 10000）
        float f_norm_rms_eps = 1e-5f;    // RMS 归一化 epsilon（llama.attention.layer_norm_rms_epsilon）

        // ---- 派生 getter（对齐上游 llama-hparams.cpp） ----
        // n_gqa = n_head / n_head_kv —— 每个 kv 头共享几个 q 头（GQA 组大小）
        uint32_t n_gqa() const { return n_head_kv == 0 ? 0 : n_head / n_head_kv; }
        // n_embd_k_gqa = n_embd_head_k * n_head_kv —— 所有 kv 头拼起来的 k/v 维度
        uint32_t n_embd_k_gqa() const { return n_embd_head_k * n_head_kv; }
    };

    // 从已解析的 GGUF KV 列表填出 HParams。
    // 成功返回 true 并填好 hp；缺关键字段返回 false（err 写原因）。
    bool parse_hparams(const gguf::gguf_context &ctx, HParams &hp, std::string &err);

} // namespace llama
