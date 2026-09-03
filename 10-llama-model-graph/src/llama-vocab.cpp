// llama-vocab.cpp - 07 章「词表 Vocab」：build(建表)、detokenize(SPM 文本直接输出、byte token 还原 1 字节)、tokenize(无 merges 逐 token 最长前缀匹配)。
// tinybrainbot 实测(SPM 型, n_vocab=32000)：model 为 "llama"；token_type NORMAL=31731/BYTE=256/USER_DEFINED=9/CONTROL=3/UNKNOWN=1；空格用 U+2581(▁)；bos=0,eos=7,unk=3,pad=2。

#include "llama-vocab.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "gguf.h"

namespace llama
{
    namespace
    {
        // 在 kv 列表里找 key（返回该条，找不到返回 nullptr）
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

        // 读一个非数组标量 u32（gguf.cpp 把标量存成十进制字符串）
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

        // 读一个 ARRAY<T> 的全部元素（元素以十进制字符串形式存于 arr_value）
        bool read_arr(const gguf::gguf_context &ctx, const std::string &key, std::vector<std::string> &out)
        {
            const gguf::gguf_kv *kv = find_key(ctx, key);
            if (kv == nullptr || !kv->is_array())
            {
                return false;
            }
            out = kv->arr_value;
            return true;
        }

        // <0xXX> 型 byte token：取第 3-4 个字符的十六进制还原成字节
        bool parse_byte_token(const std::string &t, uint8_t &b)
        {
            if (t.size() >= 6 && t[0] == '<' && t[1] == '0' && t[2] == 'x' && t[t.size() - 1] == '>')
            {
                b = (uint8_t)std::strtoul(t.substr(3, 2).c_str(), nullptr, 16);
                return true;
            }
            return false;
        }
    } // namespace

    bool Vocab::build(const gguf::gguf_context &ctx, std::string &err)
    {
        // 校验是 SPM 型（"llama"）；其他 model 本教程不支持
        const gguf::gguf_kv *m = find_key(ctx, "tokenizer.ggml.model");
        if (m == nullptr || m->value != "llama")
        {
            err = "不支持的非 llama(SPM) 词表: " + (m ? m->value : "<缺失>");
            return false;
        }

        std::vector<std::string> tok, score, type;
        if (!read_arr(ctx, "tokenizer.ggml.tokens", tok))
        {
            err = "缺少 tokenizer.ggml.tokens";
            return false;
        }
        read_arr(ctx, "tokenizer.ggml.scores", score);

        // token_type 对应上游 llama_vocab_token_type：0 UNDEFINED/1 NORMAL/2 UNKNOWN/3 CONTROL/4 USER_DEFINED/5 UNUSED/6 BYTE。
        read_arr(ctx, "tokenizer.ggml.token_type", type);

        n_vocab = (int32_t)tok.size();
        tokens.resize(n_vocab);
        token_to_id.reserve(n_vocab);
        for (int32_t i = 0; i < n_vocab; ++i)
        {
            Token &tk = tokens[i];
            tk.text = tok[i];
            tk.score = 0.0f;
            if (i < (int32_t)score.size())
            {
                tk.score = std::strtof(score[i].c_str(), nullptr);
            }
            if (i < (int32_t)type.size())
            {
                switch (std::strtol(type[i].c_str(), nullptr, 10))
                {
                case 2:
                    tk.attr = ATTR_UNKNOWN;
                    break; // UNKNOWN
                case 3:
                    tk.attr = ATTR_CONTROL;
                    break; // CONTROL
                case 4:
                    tk.attr = ATTR_USER_DEFINED;
                    break; // USER_DEFINED
                case 5:
                    tk.attr = ATTR_UNUSED;
                    break; // UNUSED
                case 6:
                    tk.attr = ATTR_BYTE;
                    break; // BYTE
                default:
                    tk.attr = ATTR_NORMAL;
                    break; // NORMAL / UNDEFINED
                }
            }
            // 反向表：text -> id（tokenize 最长前缀匹配用）
            if (token_to_id.find(tk.text) == token_to_id.end())
            {
                token_to_id[tk.text] = i;
            }
        }

        // 特殊 token id
        read_u32(ctx, "tokenizer.ggml.bos_token_id", (uint32_t &)id_bos);
        read_u32(ctx, "tokenizer.ggml.eos_token_id", (uint32_t &)id_eos);
        read_u32(ctx, "tokenizer.ggml.unknown_token_id", (uint32_t &)id_unk);
        read_u32(ctx, "tokenizer.ggml.padding_token_id", (uint32_t &)id_pad);
        // add_bos/add_eos 可缺失（实测全 0）；read_u32 失败即保持默认 false
        uint32_t t = 0;
        if (read_u32(ctx, "tokenizer.ggml.add_bos_token", t))
        {
            add_bos = t != 0;
        }
        if (read_u32(ctx, "tokenizer.ggml.add_eos_token", t))
        {
            add_eos = t != 0;
        }
        return true;
    }

    uint8_t Vocab::token_to_byte(int32_t id) const
    {
        if (id < 0 || id >= n_vocab)
        {
            return 0;
        }
        uint8_t b;
        return parse_byte_token(tokens[id].text, b) ? b : (uint8_t)0;
    }

    int32_t Vocab::byte_to_token(uint8_t ch) const
    {
        // byte token 文本是"大写"十六进制，如 <0xE4>（见 GGUF 实测），
        // token_to_id 的 key 即该原文，故用 %02X（大写）构造一致。
        char buf[8];
        std::snprintf(buf, sizeof(buf), "<0x%02X>", ch);
        auto it = token_to_id.find(buf);
        return it == token_to_id.end() ? -1 : it->second;
    }

    bool Vocab::is_eog(int32_t id) const
    {
        return id == id_eos || id == id_bos;
    }

    std::string Vocab::detokenize(const std::vector<int32_t> &ids, bool remove_special) const
    {
        std::string out;
        for (int32_t id : ids)
        {
            if (id < 0 || id >= n_vocab)
            {
                continue;
            }
            const Token &tk = tokens[id];
            if (tk.attr & ATTR_BYTE)
            {
                // <0xXX> 字节 token -> 还原成 1 个原始字节
                out += (char)token_to_byte(id);
                continue;
            }
            if (tk.attr & (ATTR_CONTROL | ATTR_USER_DEFINED | ATTR_UNKNOWN))
            {
                if (!remove_special)
                {
                    out += tk.text; // 保留特殊/未知 token 原文
                }
                continue;
            }
            // NORMAL token：SPM 文本直接输出（含 ▁ 空格占位）
            out += tk.text;
        }

        // 统一把 SPM 空格 ▁(U+2581, E2 96 81) 替换成普通空格
        std::string res;
        res.reserve(out.size());
        for (size_t i = 0; i < out.size(); ++i)
        {
            if ((unsigned char)out[i] == 0xE2 && i + 2 < out.size() &&
                (unsigned char)out[i + 1] == 0x96 && (unsigned char)out[i + 2] == 0x81)
            {
                res += ' '; // ▁ -> 空格
                i += 2;     // 跳过 3 字节里后 2 个
            }
            else
            {
                res += out[i];
            }
        }
        return res;
    }

    std::vector<int32_t> Vocab::tokenize(const std::string &text, bool add_special) const
    {
        std::vector<int32_t> out;
        if (add_special && add_bos)
        {
            out.push_back(id_bos);
        }

        // SPM：把普通空格替换成 ▁(U+2581) —— 词表里的空格 token 用 U+2581 表示
        std::string spm_text;
        spm_text.reserve(text.size());
        for (char c : text)
        {
            if (c == ' ')
            {
                spm_text += "\xE2\x96\x81"; // ▁
            }
            else
            {
                spm_text += c;
            }
        }
        // 无 merges：逐 token 最长前缀匹配（命中即取）。词表外字符按 UTF-8 逐字节编码成 <0xXX> byte token（对齐 resegment），
        // 连 byte token 都没有才回退 unk。
        for (size_t i = 0; i < spm_text.size();)
        {
            int32_t matched_id = -1;
            size_t matched_len = 0;
            // 贪心找最长命中前缀
            for (size_t l = spm_text.size() - i; l >= 1; --l)
            {
                auto it = token_to_id.find(spm_text.substr(i, l));
                if (it != token_to_id.end())
                {
                    matched_id = it->second;
                    matched_len = l;
                    break;
                }
            }
            if (matched_id >= 0 && matched_len > 0)
            {
                out.push_back(matched_id);
                i += matched_len;
                continue;
            }
            // 无命中：取当前 UTF-8 字符（首字节定字节数 1~4），逐字节编码 byte token。首字节高位前缀：0xxxxxxx=1 字节(ASCII)、110xxxxx=2(U+0080~07FF)、1110xxxx=3(U+0800~FFFF)、11110xxx=4(U+10000~10FFFF)。
            size_t clen = 1;
            unsigned char c0 = (unsigned char)spm_text[i];
            if ((c0 & 0xE0) == 0xC0)
            {
                clen = 2;
            }
            else if ((c0 & 0xF0) == 0xE0)
            {
                clen = 3;
            }
            else if ((c0 & 0xF8) == 0xF0)
            {
                clen = 4;
            }
            for (size_t k = 0; k < clen && i + k < spm_text.size(); ++k)
            {
                int32_t bt = byte_to_token((uint8_t)spm_text[i + k]);
                if (bt >= 0)
                {
                    out.push_back(bt);
                }
                else if (id_unk >= 0)
                {
                    out.push_back(id_unk);
                }
            }
            i += clen; // 推进整个 UTF-8 字符，避免死循环
            // （旧的逐字节编码方案已废弃并注释，现按上方 UTF-8 变长判断逐字节编码）
        }

        if (add_special && add_eos)
        {
            out.push_back(id_eos);
        }
        return out;
    }

} // namespace llama
