// test-llama-model-vocab.cpp - 07 章「词表 Vocab」手写测试
// 约定（见 AGENTS.md）：手写 main，退出码非 0=失败。只测词表：load_model 已 build 好 llm.vocab，断言 n_vocab/特殊 id/is_eog/tokenize<->detokenize 往返。

#include "llama-model.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace
{
    int fail(const char *msg)
    {
        std::fprintf(stderr, "FAIL %s\n", msg);
        return 1;
    }

    // 断言一个布尔条件，失败则打印并返回 false
    bool check(bool cond, const char *what)
    {
        if (!cond)
        {
            std::fprintf(stderr, "  [x] %s\n", what);
            return false;
        }
        std::printf("  [ok] %s\n", what);
        return true;
    }
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "用法: %s <model.gguf>\n", argv[0]);
        return 2;
    }

    llama::llama_model llm;
    std::string err;
    if (!llama::load_model(argv[1], llm, err))
    {
        return fail(err.c_str());
    }

    // ---- 词表 Vocab（load_model 已 build） ----
    std::printf("vocab:\n");
    const llama::Vocab &vocab = llm.vocab;
    bool ok = true;
    ok &= check(vocab.n_vocab == 32000, "n_vocab == 32000");
    ok &= check(vocab.id_bos == 0, "id_bos == 0 (<s>)");
    ok &= check(vocab.id_eos == 7, "id_eos == 7 (<|end|>)");
    ok &= check(vocab.id_unk == 3, "id_unk == 3 (<unk>)");
    ok &= check(vocab.id_pad == 2, "id_pad == 2 (<pad>)");
    ok &= check(!vocab.add_bos && !vocab.add_eos, "add_bos/add_eos 均 false（实测）");

    // is_eog：eos/bos 算句末，普通词不算
    ok &= check(vocab.is_eog(7), "is_eog(7)==true（eos）");
    ok &= check(vocab.is_eog(0), "is_eog(0)==true（bos）");
    ok &= check(!vocab.is_eog(100), "is_eog(100)==false（普通词）");

    // 演示：中文 + 英文混排的 tokenize 结果（中文无词表 token，被拆成 byte token）
    std::printf("tokenize 演示 '你好 Hello':\n");
    for (int32_t id : vocab.tokenize("你好 Hello", false))
    {
        std::printf("  id=%d text=[%s] attr=%d\n", id, vocab.tokens[id].text.c_str(), vocab.tokens[id].attr);
    }

    // tokenize -> detokenize 往返（中文 + 英文混合，应经 byte fallback 无损还原）
    const std::string src = R"(滚滚长江东逝水，浪花淘尽英雄。是非成败转头空。
　　青山依旧在，几度夕阳红。　　白发渔樵江渚上，惯
　　看秋月春风。一壶浊酒喜相逢。古今多少事，都付
　　笑谈中。 ZZZ)";
    const std::vector<int32_t> ids = vocab.tokenize(src, false);
    const std::string round = vocab.detokenize(ids, false);
    ok &= check(round == src, "tokenize->detokenize 往返一致");
    ok &= check(!ids.empty(), "tokenize 产出非空 token 序列");

    // detokenize remove_special：控制/特殊 token 被丢弃
    std::vector<int32_t> with_special;
    with_special.push_back(vocab.id_bos);
    for (int32_t id : vocab.tokenize("hi", false))
    {
        with_special.push_back(id);
    }
    with_special.push_back(vocab.id_eos);
    ok &= check(vocab.detokenize(with_special, true) == "hi",
                "detokenize(remove_special=true) 丢弃 bos/eos 特殊 token");
    if (!ok)
    {
        return fail("词表 Vocab 与 tinybrainbot 不符");
    }

    std::printf("PASS 07-llama-model-vocab\n");
    return 0;
}
