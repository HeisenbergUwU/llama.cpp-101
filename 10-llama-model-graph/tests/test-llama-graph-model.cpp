// test-llama-graph-model.cpp - 10 章「整体模型图」手写测试：加载 tinybrainbot→assemble→build_model_graph 建整图→
// compute 跑 logits→校验（大小/有限/非全同/图结构）+ 末 token argmax（与 08 直写前向 "Paris" 对齐）。算子/图设施单测在 test-ggml-graph.cpp。

#include "llama-graph.h"
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

    // ---- 加载 + 组装成语义模型 ----
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

    // ---- 提示 -> tokenize -> 位置 ----
    const std::string prompt = "The capital of France is";
    const std::vector<int32_t> toks = llm.vocab.tokenize(prompt, false);
    CHECK(!toks.empty(), "tokenize 产出非空 token 序列");
    const int N = (int)toks.size();
    std::vector<int32_t> pos(N);
    for (int i = 0; i < N; ++i)
    {
        pos[i] = i; // 绝对位置 0..N-1
    }
    std::printf("  提示 [%s] -> %d 个 token（首个 id=%d）\n", prompt.c_str(), N, toks[0]);

    // ---- 池子：no_alloc=false，算子节点 data 进池。大小按「中间张量+节点结构」保守估算（非文件大小）。
    // tinybrainbot 12 层：每层 [768,N]/[2048,N]/[256,N] + 逐头小张量 + logits[32000,N]。
    const size_t mem_size = (size_t)N * hp.n_vocab * 4 // logits
                          + (size_t)hp.n_layer * (400 * 1024) // 每层激活 + 结构余量
                          + (4u << 20); // 若干静态余量
    ggml::ggml_init_params params = {};
    params.mem_size = mem_size;
    params.no_alloc = false;
    ggml::ggml_context *ctx = ggml::ggml_init(params);
    if (ctx == NULL)
    {
        std::fprintf(stderr, "FAIL 池子建不起来（mem=%zu）\n", mem_size);
        return 1;
    }

    // ---- 建整图 ----
    llama::BuiltGraph bg = llama::build_model_graph(ctx, model, hp, toks.data(), N, pos.data(), err);
    if (bg.logits == NULL)
    {
        std::fprintf(stderr, "FAIL build_model_graph: %s\n", err.c_str());
        return 1;
    }

    // 图结构：nodes 按拓扑序（logits 在末尾），leafs 含权重（token_embd 大小 32000*768）
    CHECK(bg.cgraph.nodes.back() == bg.logits, "nodes 末尾是 logits（拓扑序）");
    bool has_embd = false;
    for (auto *l : bg.cgraph.leafs)
    {
        if (l == model.token_embd)
        {
            has_embd = true;
        }
    }
    CHECK(has_embd, "leafs 含 token_embd（embed + tied lm_head 共用）");
    std::printf("  图节点数=%zu 叶子数=%zu\n", bg.cgraph.nodes.size(), bg.cgraph.leafs.size());

    // ---- 跑图 ----
    ggml::ggml_graph_compute(&bg.cgraph);

    // ---- 校验 logits：大小 / 有限 / 非全同 / 末行 argmax ----
    // ---- 校验 logits：大小 / 有限 / 非全同 / 末行 argmax ----
    const float *logits = (const float *)bg.logits->data;
    const int n_vocab = (int)hp.n_vocab;
    CHECK(bg.logits->ne[0] == n_vocab && bg.logits->ne[1] == N,
          "logits 形状 ne={n_vocab,N}（列=词表、行=token）");
    bool all_finite = true;
    for (int i = 0; i < N * n_vocab; ++i)
    {
        if (!std::isfinite(logits[i]))
        {
            all_finite = false;
            break;
        }
    }
    CHECK(all_finite, "logits 全部有限（isfinite）");

    // 非全同：每 token 行首 logit 不全相等
    {
        bool differ = false;
        const float first = logits[0];
        for (int t = 0; t < N; ++t)
        {
            if (!approx(logits[(size_t)t * n_vocab], first, 1e-3f))
            {
                differ = true;
                break;
            }
        }
        CHECK(differ, "logits 各行不全相同");
    }

    // ---- 末 token argmax + 文本（对照 08 直写前向已知结果 "Paris"） ----
    {
        const int last = N - 1;
        const float *row = logits + (size_t)last * n_vocab;
        int argmax = 0;
        for (int j = 1; j < n_vocab; ++j)
        {
            if (row[j] > row[argmax])
            {
                argmax = j;
            }
        }
        CHECK(argmax >= 0 && argmax < n_vocab, "argmax 落在 [0, n_vocab)");
        std::string text = llm.vocab.detokenize(std::vector<int32_t>{ argmax }, true);
        std::printf("  末 token argmax id=%d  text=[%s]  logit=%.4f\n", argmax, text.c_str(), row[argmax]);
        // 08 章直写前向对同一提示的已知输出是 " Paris"（id=7831）。若两版本一致，
        // 说明 build_model_graph 建出的整图数值与直写前向对齐。
        CHECK(argmax == 7831, "末 token argmax == 7831（与 08 直写前向一致：' Paris'）");
    }

    ggml::ggml_free(ctx);

    if (g_fail == 0)
    {
        std::printf("\nPASS 10-llama-model-graph（整体模型图）\n");
        return 0;
    }
    std::printf("\nFAIL: %d 项断言失败\n", g_fail);
    return 1;
}
