// llama-cli.cpp - 11 章「命令行生成文本」应用：加载 GGUF 模型 -> llama_model::generate 自回归逐字
// 生成 -> 打印。用法：llama-cli <model.gguf> [-n max_tokens] [-p prompt...]。无 KV cache，逐 token 生成。

#include "llama-model.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace
{
    void usage(const char *prog)
    {
        std::fprintf(stderr,
                     "用法: %s <model.gguf> [-n max_tokens] [-p prompt...]\n"
                     "  <model.gguf>   模型权重（相对路径引 resources/）\n"
                     "  -n max_tokens  最多生成几个新 token（默认 64）\n"
                     "  -p prompt...   给模型的提示（可带空格，拼到行尾；默认英文俗例）\n",
                     prog);
    }
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        usage(argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];

    int max_tokens = 64;
    std::string prompt = "What is the capital of China?";
    bool have_prompt = false;

    // 解析剩余参数：-n N 设 max_tokens；-p 之后的所有参数拼成 prompt（可能带空格）
    for (int i = 2; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc)
        {
            max_tokens = std::atoi(argv[++i]);
            if (max_tokens <= 0)
            {
                std::fprintf(stderr, "max_tokens 必须 > 0\n");
                return 1;
            }
        }
        else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc)
        {
            // 把 -p 之后的所有参数拼接成 prompt（参数间用空格）
            have_prompt = true;
            prompt = argv[++i];
            while (i + 1 < argc)
            {
                prompt += " ";
                prompt += argv[++i];
            }
            break; // -p 吃到行尾，结束解析
        }
        else
        {
            std::fprintf(stderr, "未知参数: %s\n", argv[i]);
            usage(argv[0]);
            return 2;
        }
    }

    // 加载 + 组装（load_model 一步完成，填入 llm.model）
    llama::llama_model llm;
    std::string err;
    if (!llama::load_model(model_path, llm, err))
    {
        std::fprintf(stderr, "加载失败: %s\n", err.c_str());
        return 1;
    }
    std::printf("模型: %s  (n_layer=%u n_embd=%u n_vocab=%u %s)\n",
                llm.path.c_str(),
                llm.hparams.n_layer, llm.hparams.n_embd, llm.hparams.n_vocab,
                have_prompt ? "" : "[默认提示]");

    // 自回归生成，逐字打印
    llama::Sampler greedy;
    llama::Generation gen;
    std::printf("<用户> %s\n<助手> ", prompt.c_str());
    std::fflush(stdout);
    if (!llm.generate(prompt, max_tokens, greedy, gen, err))
    {
        std::fprintf(stderr, "\n生成失败: %s\n", err.c_str());
        return 1;
    }
    std::printf("\n\n[新生成 %d 个 token，%s]\n",
                gen.n_generated, gen.stopped_eog ? "命中 EOG 提前停" : "到 max_tokens 截断");

    return 0;
}
