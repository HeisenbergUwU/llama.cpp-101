// test-ggml-graph.cpp - 09 章「ggml 计算图设施」单测
//
// 不依赖模型/forward，直接在 03 章池子（ggml_init + ggml_new_tensor）上建图：
//   - 用 ggml_mul_mat / ggml_add 等从池子 new 算子节点 tensor（对齐上游）
//   - ggml_build_forward_expand 的后序去重：共享节点(m)只在 nodes 登记一次
//   - nodes 的顺序 = 拓扑序（src 先于消费者）
//   - ggml_graph_compute 按拓扑执行，数值与手算一致
//
// 约定（AGENTS.md）：手写 main，退出码非 0 = 失败。

#include "ggml-graph.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace
{
    int g_fail = 0;
#define CHECK(cond, msg)                                         \
    do                                                           \
    {                                                            \
        if (!(cond))                                             \
        {                                                        \
            std::printf("[FAIL] %s (line %d)\n", msg, __LINE__); \
            ++g_fail;                                            \
        }                                                        \
        else                                                     \
        {                                                        \
            std::printf("[ok]   %s\n", msg);                     \
        }                                                        \
    } while (0)

    bool approx(float a, float b, float tol = 1e-4f)
    {
        return std::fabs(a - b) <= tol;
    }

    // 池子里建一个 F32 叶子 tensor（ne0=最内维/列，ne1=行），并把 data 填成 arr。
    ggml::ggml_tensor *make_leaf(ggml::ggml_context *ctx, int64_t ne0, int64_t ne1,
                                 const std::vector<float> &arr, const char *name)
    {
        const int64_t ne[2] = {ne0, ne1};
        ggml::ggml_tensor *t = ggml::ggml_new_tensor(ctx, ggml::GGML_TYPE_F32, 2, ne);
        for (size_t i = 0; i < arr.size(); i++)
        {
            ((float *)t->data)[i] = arr[i];
        }
        ggml::ggml_set_name(t, name);
        return t;
    }
}

int main()
{
    using namespace ggml;

    // 池子：no_alloc=false，tensor 数据进池子（对齐上游「数据绑池子」）
    ggml_init_params params = {};
    params.mem_size  = 1u << 16; // 64KB，够这批小 tensor
    params.no_alloc  = false;
    ggml_context *ctx = ggml_init(params);
    if (ctx == NULL)
    {
        std::printf("[FAIL] 池子建不起来\n");
        return 1;
    }

    // ---- toy：一个共享节点被用两次（演示去重）----
    //   x=[1 2]
    //   W1 = |1 0|   W2 = |0 1|   （行主序，ne={n_out=2, n_in=2}）
    //        |0 1|        |1 0|
    //   m = x·W1 = [1 2]        (matmul)
    //   p = x·W2 = [2 1]
    //   a = m + p = [3 3]        (add，用 m)
    //   b = m·W2 = [2 1]        (matmul，又用 m -> m 被两个消费者引用)
    //   y = a + b = [5 4]        (add)
    //
    // 期望：build(y) 后 leafs={x,W1,W2}，nodes 里 m 只登记一次。

    ggml_tensor *x  = make_leaf(ctx, 2, 1, {1, 2}, "x");
    ggml_tensor *w1 = make_leaf(ctx, 2, 2, {1, 0, 0, 1}, "W1"); // 单位阵
    ggml_tensor *w2 = make_leaf(ctx, 2, 2, {0, 1, 1, 0}, "W2"); // 交换阵

    ggml_tensor *m = ggml_mul_mat(ctx, x, w1); // [1x2]
    ggml_tensor *p = ggml_mul_mat(ctx, x, w2); // [1x2]
    ggml_tensor *a = ggml_add(ctx, m, p);      // [1x2]
    ggml_tensor *b = ggml_mul_mat(ctx, m, w2); // [1x2]  <- m 第二次被用
    ggml_tensor *y = ggml_add(ctx, a, b);      // [1x2]

    ggml_cgraph cg;
    ggml_build_forward_expand(&cg, y);

    // ---- 1) 叶子 vs 节点 ----
    CHECK(cg.leafs.size() == 3, "leafs 数 == 3（x、W1、W2）");
    bool has_x = std::find(cg.leafs.begin(), cg.leafs.end(), x) != cg.leafs.end();
    CHECK(has_x, "leafs 包含 x");
    bool has_w1 = std::find(cg.leafs.begin(), cg.leafs.end(), w1) != cg.leafs.end();
    bool has_w2 = std::find(cg.leafs.begin(), cg.leafs.end(), w2) != cg.leafs.end();
    CHECK(has_w1 && has_w2, "leafs 包含 W1 与 W2");

    // ---- 2) 节点数与共享去重 ----
    // 节点应为 5 个：matmul(m), matmul(p), add(a), matmul(b), add(y)
    CHECK(cg.nodes.size() == 5, "nodes 数 == 5（m、p、a、b、y）");
    int count_m = (int)std::count(cg.nodes.begin(), cg.nodes.end(), m);
    CHECK(count_m == 1, "共享节点 m 在 nodes 只登记一次（去重）");

    // ---- 3) 拓扑序：src 先于消费者 ----
    CHECK(cg.nodes.back() == y, "nodes 最后一个是输出 y");
    auto pos_m = std::find(cg.nodes.begin(), cg.nodes.end(), m);
    auto pos_a = std::find(cg.nodes.begin(), cg.nodes.end(), a);
    auto pos_b = std::find(cg.nodes.begin(), cg.nodes.end(), b);
    CHECK(pos_m < pos_a, "m 先于 a（拓扑序）");
    CHECK(pos_m < pos_b, "m 先于 b（拓扑序）");

    // ---- 4) compute 数值 ----
    ggml_graph_compute(&cg);
    const float *yd = (const float *)y->data;
    const float *md = (const float *)m->data;
    const float *pd = (const float *)p->data;
    const float *ad = (const float *)a->data;
    const float *bd = (const float *)b->data;
    CHECK(approx(yd[0], 5.0f) && approx(yd[1], 4.0f),
          "compute 后 y = [5, 4]（手算一致）");
    CHECK(approx(md[0], 1.0f) && approx(md[1], 2.0f),
          "compute 后 m = x·W1 = [1, 2]");
    CHECK(approx(pd[0], 2.0f) && approx(pd[1], 1.0f),
          "compute 后 p = x·W2 = [2, 1]");
    CHECK(approx(ad[0], 3.0f) && approx(ad[1], 3.0f),
          "compute 后 a = m+p = [3, 3]");
    CHECK(approx(bd[0], 2.0f) && approx(bd[1], 1.0f),
          "compute 后 b = m·W2 = [2, 1]");

    // ---- 5) 独立性：再次 build 同一输出应重置（从空重新登记）----
    ggml_build_forward_expand(&cg, y);
    CHECK(cg.nodes.size() == 5, "重复 build 后 nodes 仍 5（build 会重置）");

    ggml_free(ctx);

    if (g_fail == 0)
    {
        std::printf("\nPASS ggml 计算图设施测试\n");
        return 0;
    }
    std::printf("\nFAIL: %d 项断言失败\n", g_fail);
    return 1;
}
