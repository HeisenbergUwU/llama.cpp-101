// graph_demo.cpp - 迷你计算图：一个文件讲清「先建图、后执行」
//
// 对应 llama.cpp 的真实分层：
//   - ggml_new_graph          （建一个空 ggml_cgraph，ggml.c）
//   - ggml_build_forward_expand（把某个张量的依赖递归登记进图，ggml.c）
//   - ggml_backend_sched_graph_compute / ggml_graph_compute（按拓扑执行，llama-context.cpp）
//
// 本文件刻意自包含（不依赖项目其他章节、不依赖真实模型权重、无三方库）。
// 只有一个概念被「挖深」：把「计算」从「执行」里剥出来。
//   - 先 build：声明「谁依赖谁、谁多大、是什么算子」-> 得到一张 DAG
//   - 再 compute：按拓扑顺序把算子一个个算掉
// 好处（详情见 README）：内存复用、跨设备、自回归增量、反向传播，都依赖「先把
// 依赖图摆出来」这一步。
//
// 编译运行（C++11，无三方依赖）：
//   g++ -std=c++11 -Wall -Wextra -O2 graph_demo.cpp -o /tmp/graph_demo && /tmp/graph_demo
// 退出码 0 表示演示数值全部通过断言（手算核对过）。

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace mini
{
    // ---- 算子类型 ----
    // 一个 tensor 若是「算子节点」，op 非 NONE 且 src 非空；若是「叶子」（权重/输入），
    // op == NONE 且 src 为空。对应 llama.cpp 里「node（data 会变） vs leaf（常量）」。
    enum class Op : uint8_t
    {
        NONE,   // 叶子：权重 / 输入，数据由外部填，算图时不变
        ADD,    // 逐元素加，src: [a, b]
        MATMUL, // 矩阵乘，src: [a(mxk), b(kxn)] -> [mxn]
        RELU,   // 逐元素 relu，src: [a]
    };

    struct Graph;

    // ---- 张量 ----
    // 同时扮演两种身份：既是「算子」（op + src 记录怎么做），也是「这个算子的
    // 输出结果的存放地」（data + ne 描述结果多大）。对应 ggml 里 nodes[] 里
    // 每一个 ggml_tensor 既是算子又是输出缓冲。
    struct Tensor
    {
        Op              op  = Op::NONE;
        std::string     name;
        std::vector<Tensor*> src;  // 算子的输入（叶子为空）
        std::vector<int64_t> ne;   // 形状（行主序，ne[0] 是最内层维度）
        std::vector<float> data;   // 数据（叶子由外部填；节点由 compute 算出）
    };

    // ---- 计算图 ----
    struct Graph
    {
        std::vector<Tensor*>            owned;   // 我拥有的所有 tensor（统一析构）
        std::vector<const Tensor*>      visited; // build 时已见过的张量（去重）
        std::vector<Tensor*>            nodes;   // 节点：算子（求值顺序 = 拓扑序）
        std::vector<Tensor*>            leafs;   // 叶子：常量/输入

        ~Graph()
        {
            for (auto *t : owned) delete t;
        }

        // 建一个叶子张量（op=NONE、src 空）。ne = 形状，data = 初始数据。
        // 对应 ggml_new_tensor + ggml_set_input。
        Tensor *leaf(const std::vector<int64_t> & shape,
                     const std::vector<float>   & data,
                     const std::string & name_)
        {
            Tensor *t = new Tensor;
            t->op  = Op::NONE;
            t->ne  = shape;
            t->data = data;
            t->name = name_;
            owned.push_back(t);
            return t;
        }

        // 建一个算子节点：拿到「结果形状 + 输入 + 算子类型」，返回输出张量。
        // 注意：这里只登记「这个算子想怎么做」，并不真正的算——和 llama.cpp 里
        // ggml_mul_mat / ggml_add 一样，调用时只建节点、不计算。
        Tensor *node(Op op, const std::vector<int64_t> & shape,
                     const std::vector<Tensor*> & inputs,
                     const std::string & name_)
        {
            Tensor *t = new Tensor;
            t->op   = op;
            t->ne   = shape;
            t->src  = inputs;
            t->name = name_;
            t->data.assign(total(shape), 0.0f);
            owned.push_back(t);
            return t;
        }

        // 便捷算子 API（对应 llama.cpp 的 ggml_add / ggml_mul_mat / ggml_relu）。
        Tensor *add(Tensor *a, Tensor *b)
        {
            return node(Op::ADD, a->ne, {a, b}, a->name + "+" + b->name);
        }
        Tensor *matmul(Tensor *a, Tensor *b)
        {
            // a: [m x k], b: [k x n] -> out: [m x n]
            std::vector<int64_t> shape = {a->ne[0], b->ne[1]};
            return node(Op::MATMUL, shape, {a, b}, a->name + "@" + b->name);
        }
        Tensor *relu(Tensor *a)
        {
            return node(Op::RELU, a->ne, {a}, "relu(" + a->name + ")");
        }

        // 一个张量总共几个元素（= 每维乘起来）
        static size_t total(const std::vector<int64_t> & ne)
        {
            size_t n = 1;
            for (auto d : ne) n *= (size_t)d;
            return n;
        }

        // ---- build：把 out 的整条依赖链登记成图（对应 build_forward_expand）----
        // 从最终输出出发，递归往下：先 visit 所有输入，再把自己入图。
        // 用 visited 去重：共享的中间结果（比如被两个节点用）只登记一次。
        void build(Tensor *out)
        {
            visit(out);
        }

    private:
        void visit(Tensor *t)
        {
            // 已见过 -> 跳过，避免重复入图（也防成环死循环）
            for (auto *v : visited)
            {
                if (v == t) return;
            }
            visited.push_back(t);

            // 先递归登记所有输入（保证依赖在消费者之前入图 = 天然拓扑序）
            for (auto *s : t->src) visit(s);

            // 然后分叶子 / 节点登记
            if (t->op == Op::NONE)
            {
                leafs.push_back(t);   // 常量/权重/输入
            }
            else
            {
                nodes.push_back(t);   // 算子：nodes 的顺序 = 求值顺序
            }
        }

    public:
        // ---- compute：按 nodes 拓扑顺序逐个执行算子（对应 graph_compute）----
        // 这里每个节点读取它 src 的 data、算出自己的 data。
        void compute()
        {
            for (auto *t : nodes)
            {
                switch (t->op)
                {
                case Op::ADD:
                {
                    const auto &A = t->src[0]->data;
                    const auto &B = t->src[1]->data;
                    for (size_t i = 0; i < t->data.size(); i++) t->data[i] = A[i] + B[i];
                    break;
                }
                case Op::MATMUL:
                {
                    // A: [m x k], B: [k x n] -> C: [m x n]
                    const int64_t m = t->src[0]->ne[0];
                    const int64_t k = t->src[0]->ne[1];
                    const int64_t n = t->src[1]->ne[1];
                    const auto &A = t->src[0]->data;
                    const auto &B = t->src[1]->data;
                    for (int64_t i = 0; i < m; i++)
                    {
                        for (int64_t j = 0; j < n; j++)
                        {
                            float s = 0.0f;
                            for (int64_t q = 0; q < k; q++)
                            {
                                s += A[i*k + q] * B[q*n + j];
                            }
                            t->data[i*n + j] = s;
                        }
                    }
                    break;
                }
                case Op::RELU:
                {
                    const auto &A = t->src[0]->data;
                    for (size_t i = 0; i < t->data.size(); i++)
                    {
                        t->data[i] = A[i] > 0.0f ? A[i] : 0.0f;
                    }
                    break;
                }
                case Op::NONE:
                    // 叶子不应出现在 nodes 里
                    std::fprintf(stderr, "bug: leaf in nodes\n");
                    std::exit(1);
                }
            }
        }
    };

} // namespace mini

// ---- 取张量第 r 行 c 列的元素（行主序一维索引），打印用 ----
static float at(const mini::Tensor *t, size_t r, size_t c)
{
    return t->data[r * (size_t)t->ne[1] + c];
}

// ---- 打印一张图的拓扑：叶子 + 节点（按执行顺序）----
static void print_graph(const mini::Graph &g)
{
    std::printf("== 叶子 leafs（常量/权重/输入，不参与计算）==\n");
    for (const auto *t : g.leafs)
    {
        std::printf("  leaf  %-12s ne=[", t->name.c_str());
        for (size_t i = 0; i < t->ne.size(); i++)
            std::printf("%s%lld", i ? "," : "", (long long)t->ne[i]);
        std::printf("]\n");
    }

    std::printf("\n== 节点 nodes（算子，按此顺序执行 = 拓扑序）==\n");
    for (size_t i = 0; i < g.nodes.size(); i++)
    {
        const auto *t = g.nodes[i];
        const char *op = t->op == mini::Op::ADD    ? "add   " :
                         t->op == mini::Op::MATMUL ? "matmul" :
                         t->op == mini::Op::RELU   ? "relu  " : "?";
        std::printf("  [%02zu] %-6s %-16s <-", i, op, t->name.c_str());
        for (const auto *s : t->src)
            std::printf(" %s", s->name.c_str());
        std::printf("  ne=[%lld,%lld]\n", (long long)t->ne[0], (long long)t->ne[1]);
    }
}

// ---- 打印矩阵（用于核对数值）----
static void print_mat(const char *tag, const mini::Tensor *t)
{
    std::printf("%s [%lld x %lld]:\n", tag, (long long)t->ne[0], (long long)t->ne[1]);
    for (int64_t r = 0; r < t->ne[0]; r++)
    {
        std::printf("  ");
        for (int64_t c = 0; c < t->ne[1]; c++)
            std::printf("%8.3f ", at(t, (size_t)r, (size_t)c));
        std::printf("\n");
    }
}

int main()
{
    using namespace mini;

    Graph g;

    // ---- toy 例子：一个带「残差」的 2 层 MLP ----
    //   x ---------> h1 = x·W1 --------> h2 = relu(h1) --+
    //                 (matmul)          (relu)           |
    //                                                     +-> h3 = h2 + h1  ->  y = h3·W2
    //   权重: W1[3x4]  W2[4x2]        (h1 被 h2 和 h3 用了两次!用来演示去重)
    //
    // 手算验证:
    //   x = [1 2 3]
    //   W1 = | 1 0 1 1 |      W2 = | 1 1 |
    //        |-1 1 0 1 |           | 0 1 |
    //        | 1 1 1 0 |           | 1 0 |
    //                             | 1 1 |
    //   h1 = x·W1 = [1*1+2*-1+3*1, 1*0+2*1+3*1, 1*1+2*0+3*1, 1*1+2*1+3*0]
    //              = [2, 5, 4, 3]
    //   h2 = relu(h1) = [2, 5, 4, 3]   (全部 >0)
    //   h3 = h2 + h1  = [4, 10, 8, 6]
    //   y  = h3·W2:
    //        col0 = 4*1 + 10*0 + 8*1 + 6*1 = 18
    //        col1 = 4*1 + 10*1 + 8*0 + 6*1 = 20
    //   y = [18, 20]

    Tensor *x  = g.leaf({1, 3}, {1, 2, 3}, "x");
    Tensor *W1 = g.leaf({3, 4}, {1,0,1,1, -1,1,0,1, 1,1,1,0}, "W1");
    Tensor *W2 = g.leaf({4, 2}, {1,1, 0,1, 1,0, 1,1}, "W2");

    Tensor *h1 = g.matmul(x, W1);      // [1x4]
    Tensor *h2 = g.relu(h1);           // [1x4]
    Tensor *h3 = g.add(h2, h1);        // [1x4]  <- h1 被用两次
    Tensor *y  = g.matmul(h3, W2);     // [1x2]

    // ---- 先建图：从最终输出 y 反推整条依赖链 ----
    std::printf("### 第 1 步：build —— 从输出 y 递归登记依赖（build_forward_expand）\n\n");
    g.build(y);

    // 展示图里到底登记了什么
    print_graph(g);

    std::printf("\n注意:  h1 被 h2 和 h3 两个节点引用，但 build 去重后 nodes 里只登记一次。\n");
    std::printf("      这就是 ggml 用 visited(哈希) 去重的意义——共享子图不重复登记。\n");

    // ---- 再执行：按拓扑顺序逐个算子算 ----
    std::printf("\n\n### 第 2 步：compute —— 按 nodes 顺序逐个执行算子（graph_compute）\n");
    g.compute();

    print_mat("\nh1 = x·W1", h1);
    print_mat("h2 = relu(h1)", h2);
    print_mat("h3 = h2 + h1 (残差)", h3);
    print_mat("\ny  = h3·W2", y);

    // ---- 断言：手算核对 ----
    int failures = 0;
    if (std::fabs(at(y, 0, 0) - 18.0f) > 1e-5f) { std::printf("FAIL y[0] != 18\n"); failures++; }
    if (std::fabs(at(y, 0, 1) - 20.0f) > 1e-5f) { std::printf("FAIL y[1] != 20\n"); failures++; }

    std::printf("\n### 结论\n");
    if (failures == 0)
    {
        std::printf("数值全部对上（y = [18, 20]）。建图 + 执行的迷你流程演示成功，退出码 0。\n");
        return 0;
    }
    std::printf("数值断言失败 %d 处\n", failures);
    return 1;
}
