// test-llama-sample.cpp - 11 章「自回归采样 + 跑起来」手写测试（AGENTS.md：退出码非 0=失败）。加载 tinybrainbot
// ->generate 逐字打印到 EOG/max_tokens。校验：n_generated>0 且首新 token=" Paris"(id=7831) + 停止合法 + 输出非空。

#include "llama-model.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "用法: %s <model.gguf> [max_tokens] [prompt...]\n", argv[0]);
        return 2;
    }

    // ---- 加载（load_model 内部已组装成语义模型，填入 llm.model） ----
    llama::llama_model llm;
    std::string err;
    if (!llama::load_model(argv[1], llm, err))
    {
        std::fprintf(stderr, "FAIL load_model: %s\n", err.c_str());
        return 1;
    }

    // ---- 参数：max_tokens（默认 32）与 prompt（默认法式梗，与 08/10 对齐） ----
    int max_tokens = 32;
    if (argc >= 3)
    {
        max_tokens = std::atoi(argv[2]);
        if (max_tokens <= 0)
        {
            std::fprintf(stderr, "FAIL: max_tokens 必须 > 0\n");
            return 1;
        }
    }
    std::string prompt = "The capital of France is";
    bool default_prompt = true; // 是否用了默认提示（其首个新 token 与 08/10 对齐已知）
    if (argc >= 4)
    {
        default_prompt = false;
        prompt = argv[3];
        for (int i = 4; i < argc; ++i)
        {
            prompt += " ";
            prompt += argv[i];
        }
    }

    std::printf("提示 [%s]  max_tokens=%d  采样器=greedy(argmax)  无KV cache(每步全量重算)\n",
                prompt.c_str(), max_tokens);
    std::printf("生成: ");

    // ---- 自回归采样（直接挂在加载好的 llama_model 上） ----
    llama::Sampler greedy; // 默认 greedy
    llama::Generation gen;
    if (!llm.generate(prompt, max_tokens, greedy, gen, err))
    {
        std::fprintf(stderr, "\nFAIL generate: %s\n", err.c_str());
        return 1;
    }
    std::printf("\n\n");

    int fail = 0;
    #define CHECK(cond, msg)                                           \
        do                                                             \
        {                                                              \
            if (!(cond))                                               \
            {                                                          \
                std::printf("[FAIL] %s\n", msg);                       \
                ++fail;                                                \
            }                                                          \
            else                                                       \
            {                                                          \
                std::printf("[ok]   %s\n", msg);                       \
            }                                                          \
        } while (0)

    // 必然推进：至少生成了 1 个新 token 且文本非空
    CHECK(gen.n_generated > 0, "有生成新 token（n_generated > 0）");
    CHECK(gen.n_prompt > 0, "输入 prompt 已计入 token 序列");
    CHECK(!gen.text.empty(), "生成文本非空");

    // 首个新 token 是 10 章已知的 " Paris"（id=7831）——证明建图+采样数值对齐。
    // 仅对默认提示成立（别的提示首个 token 理应不同）。
    if (gen.n_generated >= 1)
    {
        const int32_t first = gen.token_ids[gen.n_prompt];
        if (default_prompt)
        {
            CHECK(first == 7831, "首个新 token id==7831（与 10 章整体模型图一致: ' Paris'）");
        }
        else
        {
            std::printf("  [info] 自定义提示：首个新 token id=%d（不做 7831 断言）\n", first);
        }
    }
    else
    {
        CHECK(false, "首个新 token 存在（n_generated>=1 才能验）");
    }

    // 停止条件：要么 EOG，要么触顶 max_tokens
    CHECK(gen.stopped_eog || gen.n_generated == max_tokens,
          "终止条件合法（EOG 或 max_tokens）");

    if (gen.stopped_eog)
    {
        std::printf("  停止: 命中 EOG（在 max_tokens 之前结束） n_generated=%d\n", gen.n_generated);
    }
    else
    {
        std::printf("  停止: 到 max_tokens=%d 截断 n_generated=%d\n", max_tokens, gen.n_generated);
    }
    std::printf("  生成文本: [%s]\n", gen.text.c_str());

    if (fail == 0)
    {
        std::printf("\nPASS 11-llama-model-run-it!（自回归采样 + 能跑起来）\n");
        return 0;
    }
    std::printf("\nFAIL: %d 项断言失败\n", fail);
    return 1;
}
