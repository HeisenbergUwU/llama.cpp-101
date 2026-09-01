// test-llama-forward.cpp - 08 章「完整前向」手写测试（外层组件）
//
// 约定（见 AGENTS.md）：手写 main，退出码非 0 = 失败；模型路径是首个参数。
// 只测"具体组件"：ForwardInput（输入载体）+ KVSlice / build_causal_mask（接缝）
// + forward（6 段前向）——加载 tinybrainbot -> tokenize 短提示 -> forward 出 logits
// -> 校验（大小 / 有限 / 非全同 / kv.n_kv）-> 末 token argmax。
//
// 算子的单独测试见 tests/test-kernel.cpp（可独立编译/跑）。

#include "llama-forward.h"
#include "llama-model.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace
{
    int g_fail = 0;
    #define CHECK(cond, msg)                                          \
        do                                                            \
        {                                                             \
            if (!(cond))                                              \
            {                                                         \
                std::printf("[FAIL] %s (line %d)\n", msg, __LINE__);  \
                ++g_fail;                                             \
            }                                                         \
            else                                                      \
            {                                                         \
                std::printf("[ok]   %s\n", msg);                      \
            }                                                         \
        } while (0)

    bool approx(float a, float b, float tol = 1e-4f)
    {
        return std::fabs(a - b) <= tol;
    }
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "用法: %s <model.gguf>\n", argv[0]);
        return 2;
    }

    // ========== 结构组件：ForwardInput / build_causal_mask ==========
    std::printf("== 组件单测（ForwardInput / build_causal_mask） ==\n");

    // ForwardInput：seq_id + 三参数 append + 等长不变式
    {
        llama::ForwardInput in = llama::ForwardInput::make_batch(3);
        in.append(10, 0, 0);
        in.append(20, 0, 1);
        in.append(11, 1, 0);
        CHECK(in.seq_id == std::vector<int32_t>({ 0, 1, 0 }), "ForwardInput append(token,pos,seq_id)");
        CHECK(in.tokens.size() == in.pos.size() && in.pos.size() == in.seq_id.size(),
              "ForwardInput 三数组等长不变式");
        CHECK(llama::ForwardInput::from_tokens({ 1, 2 }, 7).pos == std::vector<int32_t>({ 7, 8 }),
              "from_tokens 绝对位置从 seq_start 递增");
    }

    // build_causal_mask：s<=t -> 0；s>t -> -inf
    {
        const int n = 4;
        std::vector<float> m = llama::build_causal_mask(n);
        bool ok = true;
        for (int t = 0; t < n; ++t)
        {
            for (int s = 0; s < n; ++s)
            {
                if (s <= t)
                {
                    if (!approx(m[(size_t)t * n + s], 0.0f, 1e-9f))
                    {
                        ok = false;
                        std::printf("[FAIL] mask[%d][%d] 应为 0\n", t, s);
                    }
                }
                else if (m[(size_t)t * n + s] != -INFINITY)
                {
                    ok = false;
                    std::printf("[FAIL] mask[%d][%d] 应为 -inf\n", t, s);
                }
            }
        }
        CHECK(ok, "build_causal_mask 因果（s<=t 为 0，s>t 为 -inf）");
    }

    // ========== 模型前向 ==========
    std::printf("\n== 模型前向（tinybrainbot） ==\n");

    llama::llama_model llm;
    std::string err;
    if (!llama::load_model(argv[1], llm, err))
    {
        std::fprintf(stderr, "FAIL load_model: %s\n", err.c_str());
        return 1;
    }

    llama::Model model;
    if (!llama::assemble_model(llm, model, err))
    {
        std::fprintf(stderr, "FAIL assemble_model: %s\n", err.c_str());
        return 1;
    }

    const llama::HParams &hp = llm.hparams;
    CHECK(hp.n_embd == 768 && hp.n_layer == 12 && hp.n_head == 12 && hp.n_head_kv == 4 &&
          hp.n_ff == 2048 && hp.n_vocab == 32000 && hp.n_rot == 64,
          "hparams 实测值（n_embd=768/n_layer=12/n_head=12/n_head_kv=4/n_ff=2048/n_vocab=32000/n_rot=64）");
    CHECK(hp.n_gqa() == 3, "n_gqa == n_head/n_head_kv == 3");
    CHECK(model.output == model.token_embd, "output tied（复用 token_embd）");
    CHECK(model.layers.size() == hp.n_layer, "model.layers.size() == n_layer");

    // 短提示 -> tokenize（SPM；add_special=false 无 bos/eos）
    const std::string prompt = "The capital of France is";
    const std::vector<int32_t> toks = llm.vocab.tokenize(prompt, false);
    CHECK(!toks.empty(), "tokenize 产出非空 token 序列");
    std::printf("  提示 [%s] -> %zu 个 token（首个 id=%d）\n", prompt.c_str(), toks.size(), toks[0]);

    llama::ForwardInput input = llama::ForwardInput::from_tokens(toks, 0);
    const int N = (int)input.n_tokens();

    llama::KVSlice kv;
    std::vector<float> logits;
    if (!llama::forward(model, hp, input, kv, logits, err))
    {
        std::fprintf(stderr, "FAIL forward: %s\n", err.c_str());
        return 1;
    }

    // -- 校验 logits -- 大小 / 有限 / 非全同
    CHECK(logits.size() == (size_t)N * hp.n_vocab, "logits 大小 == n_tokens × n_vocab");
    bool all_finite = true;
    for (float lg : logits)
    {
        if (!std::isfinite(lg))
        {
            all_finite = false;
            break;
        }
    }
    CHECK(all_finite, "logits 全部有限（isfinite）");

    // 非全同：取每个 token 行的首个 logit，看是否都相等
    {
        bool differ = false;
        float first = logits[0];
        for (int t = 0; t < N; ++t)
        {
            if (!approx(logits[(size_t)t * hp.n_vocab], first, 1e-3f))
            {
                differ = true;
                break;
            }
        }
        CHECK(differ, "logits 各行不全相同");
    }

    CHECK(kv.n_kv == N, "forward 后 kv.n_kv == n_tokens");

    // -- 末 token argmax + 打印其 id 与文本 --
    {
        const int last = N - 1;
        const float *row = &logits[(size_t)last * hp.n_vocab];
        int argmax = 0;
        for (int j = 1; j < (int)hp.n_vocab; ++j)
        {
            if (row[j] > row[argmax])
            {
                argmax = j;
            }
        }
        CHECK(argmax >= 0 && argmax < (int)hp.n_vocab, "argmax 落在 [0, n_vocab)");
        std::string text = llm.vocab.detokenize(std::vector<int32_t>{ argmax }, true);
        std::printf("  末 token argmax id=%d  text=[%s]  logit=%.4f\n", argmax, text.c_str(), row[argmax]);
        // 打印前几个候选，直观展示 logits 有区分度
        std::printf("  top-3 候选：");
        std::vector<int> idx((int)hp.n_vocab);
        for (int j = 0; j < (int)hp.n_vocab; ++j)
        {
            idx[j] = j;
        }
        for (int a = 0; a < 3; ++a)
        {
            for (int b = a + 1; b < (int)hp.n_vocab; ++b)
            {
                if (row[idx[b]] > row[idx[a]])
                {
                    std::swap(idx[a], idx[b]);
                }
            }
            std::string tt = llm.vocab.detokenize(std::vector<int32_t>{ idx[a] }, true);
            std::printf("[%d '%s' %.3f] ", idx[a], tt.c_str(), row[idx[a]]);
        }
        std::printf("\n");
    }

    if (g_fail == 0)
    {
        std::printf("\nPASS 08-llama-model-kernel（完整前向）\n");
        return 0;
    }
    std::printf("\nFAIL: %d 项断言失败\n", g_fail);
    return 1;
}
