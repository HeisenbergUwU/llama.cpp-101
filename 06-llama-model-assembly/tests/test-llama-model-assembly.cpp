// test-llama-model-layer.cpp - 06 章「模型语义：Layer 组装」的手写测试
//
// 约定（见 AGENTS.md）：手写 main，退出码非 0 = 失败。
// 流程：load_model（内部已 parse_hparams 填好 llm.hparams）
//     -> assemble_model 把 110 个张量挂载成语义结构
//     -> 断言 HParams 实测值 + 层数/指针完备性/形状。

#include "llama-hparams.h"
#include "llama-model.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

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

    // ---- 1) HParams（load_model 内部已解析，对照 tinybrainbot 实测值） ----
    std::printf("hparams:\n");
    const llama::HParams &hp = llm.hparams;
    bool ok = true;
    ok &= check(hp.n_embd == 768, "n_embd == 768");
    ok &= check(hp.n_layer == 12, "n_layer == 12");
    ok &= check(hp.n_head == 12, "n_head == 12");
    ok &= check(hp.n_head_kv == 4, "n_head_kv == 4");
    ok &= check(hp.n_ff == 2048, "n_ff == 2048");
    ok &= check(hp.n_embd_head_k == 64, "n_embd_head_k == 64");
    ok &= check(hp.n_embd_head_v == 64, "n_embd_head_v == 64");
    ok &= check(hp.n_rot == 64, "n_rot == 64");
    ok &= check(hp.n_vocab == 32000, "n_vocab == 32000");
    ok &= check(hp.n_gqa() == 3, "n_gqa()==n_head/n_head_kv == 3");
    ok &= check(hp.n_embd_k_gqa() == 256, "n_embd_k_gqa()==n_embd_head_k*n_head_kv == 256");
    ok &= check(std::fabs(hp.rope_freq_base - 10000.0f) < 1e-3f, "rope_freq_base == 10000");
    ok &= check(std::fabs(hp.f_norm_rms_eps - 1e-5f) < 1e-9f, "f_norm_rms_eps == 1e-5");
    if (!ok)
    {
        return fail("hparams 与 tinybrainbot 实测值不符");
    }

    // ---- 2) 语义组装 ----
    llama::Model model;
    if (!llama::assemble_model(llm, model, err))
    {
        return fail(err.c_str());
    }

    std::printf("semantics:\n");
    ok = true;
    ok &= check(model.token_embd != nullptr, "token_embd != NULL");
    ok &= check(model.output_norm != nullptr, "output_norm != NULL");
    ok &= check(model.output != nullptr, "output != NULL");
    // tinybrainbot 无独立 output.weight：lm_head 做权重绑定复用 token_embd
    ok &= check(model.output == model.token_embd, "output 权重绑定复用 token_embd（无独立 lm_head）");
    ok &= check(model.layers.size() == 12, "layers.size() == 12");

    for (size_t i = 0; i < model.layers.size(); i++)
    {
        const llama::Layer &L = model.layers[i];
        if (!(L.attn_norm && L.wq && L.wk && L.wv && L.wo &&
              L.ffn_norm && L.gate && L.up && L.down))
        {
            std::fprintf(stderr, "  [x] layer[%zu] 缺权重\n", i);
            return fail("存在缺权重的层");
        }
    }
    std::printf("  [ok] 每层 9 个权重齐全（attn_norm/wq/wk/wv/wo/ffn_norm/gate/up/down）\n");

    const llama::Layer &L0 = model.layers[0];
    ok &= check(L0.wq->ne[1] == 768, "wq->ne[1] == 768 (n_embd)");
    ok &= check(L0.wk->ne[1] == 256, "wk->ne[1] == 256 (n_embd_k_gqa)");
    ok &= check(L0.wv->ne[1] == 256, "wv->ne[1] == 256 (n_embd_k_gqa)");
    ok &= check(L0.wo->ne[1] == 768, "wo->ne[1] == 768 (n_embd)");
    ok &= check(L0.gate->ne[1] == 2048, "gate->ne[1] == 2048 (n_ff)");
    ok &= check(L0.up->ne[1] == 2048, "up->ne[1] == 2048 (n_ff)");
    ok &= check(L0.down->ne[0] == 2048, "down->ne[0] == 2048 (n_ff)");
    ok &= check(model.token_embd->ne[1] == 32000, "token_embd->ne[1] == 32000 (n_vocab)");
    if (!ok)
    {
        return fail("语义形状与 tinybrainbot 不符");
    }

    std::printf("PASS 06-llama-model-layer\n");
    return 0;
}
