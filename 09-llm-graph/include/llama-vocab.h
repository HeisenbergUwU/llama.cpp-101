// llama-vocab.h - 07 章「词表 Vocab」接口（最小可跑）
//
// 从 GGUF 的 tokenizer.ggml.* KV 建词表，提供 id<->文本 编解码。
// 教学取舍（对照上游 llama.cpp/src/llama-vocab.h/.cpp）：
//   - 上游 llama_vocab 用 pimpl（struct impl + unique_ptr）藏实现、从
//     llama_model_loader + LLM_KV 读 KV、按 model 派发 SPM/BPE/WPM/... 几十种。
//   - 本迷你版：扁平 struct、直接读 gguf_context.kv、只支持 tinybrainbot
//     （tokenizer.ggml.model="llama" 即 SPM 型，无 merges）。
// 故不引入上游的 llama_vocab_pre_type 全表、normalizer_options 等 -- 最小集够用。

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace gguf
{
    struct gguf_context; // build() 只读它的 kv 列表
}

namespace llama
{

    // 词表：id -> token 描述 + 特殊 token id + 编解码接口
    struct Vocab
    {
        enum Attr
        {
            ATTR_NORMAL = 1 << 0,
            ATTR_UNKNOWN = 1 << 1,
            ATTR_CONTROL = 1 << 2,
            ATTR_USER_DEFINED = 1 << 3,
            ATTR_UNUSED = 1 << 4,
            ATTR_BYTE = 1 << 5,
        };

        // 一个 token 描述（对应上游 llama_vocab::token_data）
        struct Token
        {
            std::string text; // 原始文本
            float score = 0.0f;
            int attr = ATTR_NORMAL; // tinybrainbot 用 NORMAL/CONTROL/BYTE 等
        };

        // ---- 数据成员（上游在 pimpl 里，此处扁平直接暴露） ----
        std::vector<Token> tokens;                            // id -> token（长度 = n_vocab）
        std::unordered_map<std::string, int32_t> token_to_id; // 反向表：text -> id（tokenize 用）
        int32_t n_vocab = 0;

        // 特殊 token id（tinybrainbot 实测：bos=0, eos=7, unk=3, pad=2）
        int32_t id_bos = 0;
        int32_t id_eos = 0;
        int32_t id_unk = 0;
        int32_t id_pad = 0;
        bool add_bos = false; // tokenize 时是否前置 bos（实测 false）
        bool add_eos = false; // tokenize 时是否末尾追加 eos（实测 false）

        // ---- 接口 ----
        // 从 GGUF KV 建词表（tokenizer.ggml.tokens/scores/token_type/特殊 id）。
        // 成功 true 填好本对象；失败 false（err 写原因）。
        bool build(const gguf::gguf_context &ctx, std::string &err);

        // <0xXX> 型 byte token 解析成 1 个字节；非 byte token 返回 0
        uint8_t token_to_byte(int32_t id) const;

        // 1 个字节 -> 对应 <0xXX> byte token 的 id；词表无该 byte 返回 -1
        // （tokenize 的词表外编码兜底，对齐上游 llama_vocab::byte_to_token）
        int32_t byte_to_token(uint8_t ch) const;

        // 是否 EOG（eos/bos 等"句末"token）—— 自回归采样章的停止条件
        bool is_eog(int32_t id) const;

        // id 序列 -> 文本。remove_special=true 丢 CONTROL/特殊 token 原文。
        // 对 SPM：normal token 直接 append text，byte token 还原成 1 字节。
        std::string detokenize(const std::vector<int32_t> &ids, bool remove_special) const;

        // 文本 -> id 序列。无 merges，做逐 token 最长前缀匹配（token_to_id 命中即取），
        // 找不到回退 unk；add_special 时按需前置 bos / 追加 eos。
        std::vector<int32_t> tokenize(const std::string &text, bool add_special) const;
    };

} // namespace llama
