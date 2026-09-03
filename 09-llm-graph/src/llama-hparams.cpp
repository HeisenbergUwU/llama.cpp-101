// llama-hparams.cpp - 06 章「超参数 HParams」实现
// parse_hparams：遍历 gguf_context.kv，按 key 把值填进 HParams。键名="<arch>.<subkey>"（arch 对 plain llama 是 "llama"，见 llama-arch.cpp）；只处理 tinybrainbot 需要的字段，缺关键字段即报错（读者可在此扩展）。

#include "llama-hparams.h"

#include <cstdlib>
#include <string>
#include <vector>

#include "gguf.h"

namespace llama
{
    namespace
    {
        // 在 kv 列表里找 key，返回该条（找不到返回 nullptr）。
        const gguf::gguf_kv *find_key(const gguf::gguf_context &ctx, const std::string &key)
        {
            for (const gguf::gguf_kv &kv : ctx.kv)
            {
                if (kv.key == key)
                {
                    return &kv;
                }
            }
            return nullptr;
        }

        // 读一个非数组、标量数值型 KV：uint32 或 float。05 的 gguf.cpp 把标量统一存成十进制字符串（std::to_string），故用 strtoul/strtof 就能还原。
        bool read_u32(const gguf::gguf_context &ctx, const std::string &key, uint32_t &out)
        {
            const gguf::gguf_kv *kv = find_key(ctx, key);
            if (kv == nullptr)
            {
                return false;
            }
            out = (uint32_t)std::strtoul(kv->value.c_str(), nullptr, 10);
            return true;
        }

        bool read_f32(const gguf::gguf_context &ctx, const std::string &key, float &out)
        {
            const gguf::gguf_kv *kv = find_key(ctx, key);
            if (kv == nullptr)
            {
                return false;
            }
            out = std::strtof(kv->value.c_str(), nullptr);
            return true;
        }
    } // namespace

    bool parse_hparams(const gguf::gguf_context &ctx, HParams &hp, std::string &err)
    {
        // 先核对架构：本教程只支持 plain llama（tinybrainbot / 后续 llama 系）。
        const gguf::gguf_kv *arch = find_key(ctx, "general.architecture");
        if (arch == nullptr || arch->value != "llama")
        {
            err = "仅支持 architecture=llama 的模型（当前: " +
                  (arch ? arch->value : std::string("<无 general.architecture>")) + "）";
            return false;
        }

        // ---- 直接读 KV（缺任何一个关键字段都算解析失败） ----
        if (!read_u32(ctx, "llama.embedding_length", hp.n_embd) ||
            !read_u32(ctx, "llama.block_count", hp.n_layer) ||
            !read_u32(ctx, "llama.attention.head_count", hp.n_head) ||
            !read_u32(ctx, "llama.feed_forward_length", hp.n_ff) ||
            !read_u32(ctx, "llama.vocab_size", hp.n_vocab))
        {
            err = "缺少必需超参数（n_embd/n_layer/n_head/n_ff/n_vocab）";
            return false;
        }

        // ---- 可选 / 带默认值的字段（对应上游 load_hparams 的 false=optional） ----
        // head_count_kv：缺省 = n_head（MHA；有则为 GQA，tinybrainbot=4）
        hp.n_head_kv = hp.n_head;
        read_u32(ctx, "llama.attention.head_count_kv", hp.n_head_kv);

        // 每头维度：默认 n_embd / n_head，可被 attention.key_length/value_length 覆盖
        hp.n_embd_head_k = hp.n_embd / hp.n_head;
        read_u32(ctx, "llama.attention.key_length", hp.n_embd_head_k);
        hp.n_embd_head_v = hp.n_embd / hp.n_head;
        read_u32(ctx, "llama.attention.value_length", hp.n_embd_head_v);

        // RoPE 维度：默认 = n_embd_head_k，可被 rope.dimension_count 覆盖
        hp.n_rot = hp.n_embd_head_k;
        read_u32(ctx, "llama.rope.dimension_count", hp.n_rot);

        // norm epsilon：默认 1e-5
        read_f32(ctx, "llama.attention.layer_norm_rms_epsilon", hp.f_norm_rms_eps);
        // rope 基频：默认 10000
        read_f32(ctx, "llama.rope.freq_base", hp.rope_freq_base);

        return true;
    }

} // namespace llama
