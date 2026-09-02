// test-kernel.cpp - 08 章「算子内核」单独测试
//
// 只测 kernel 层的纯算子（fp16_to_fp32 / dequant_row / rms_norm / matmul /
// silu / rope_inplace / softmax_row），**不依赖模型 / forward / KVSlice**：
// 可以单独编译、单独跑（CMake 的 test-kernel 目标）。
//
// 约定（AGENTS.md）：手写 main，退出码非 0 = 失败。
// 注：memset 一个本地 F32 ggml_tensor 来喂 matmul/dequant_row，不依赖池子。

#include "kernel/kernel.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
}

int main()
{
    // --- fp16_to_fp32 已知值 ---
    CHECK(approx(kernel::fp16_to_fp32(0x3C00), 1.0f), "fp16 0x3C00 == 1.0");
    CHECK(approx(kernel::fp16_to_fp32(0xBC00), -1.0f), "fp16 0xBC00 == -1.0");
    CHECK(approx(kernel::fp16_to_fp32(0x0000), 0.0f), "fp16 0x0000 == 0.0");
    CHECK(approx(kernel::fp16_to_fp32(0x4000), 2.0f), "fp16 0x4000 == 2.0");
    CHECK(approx(kernel::fp16_to_fp32(0x3C01), 1.0009765625f), "fp16 0x3C01 == 1.0009765625");

    // --- rope_inplace ---
    {
        float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        float orig[4];
        std::memcpy(orig, x, sizeof(x));
        kernel::rope_inplace(x, 4, 1, 10000.0f); // pos=1 -> 应改变第一半
        bool changed = !approx(x[0], orig[0], 1e-5f);
        CHECK(changed, "rope_inplace pos>0 改变前 n_rot/2 维");
        // 旋转保持范数：sum(x^2) 不变
        float s0 = orig[0] * orig[0] + orig[1] * orig[1] + orig[2] * orig[2] + orig[3] * orig[3];
        float s1 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
        CHECK(approx(s0, s1, 1e-3f), "rope_inplace 保持范数不变");
    }

    // --- softmax_row：行内归一，和为 1 ---
    {
        float row[4] = {1.0f, 2.0f, 3.0f, 10.0f};
        kernel::softmax_row(row, 4);
        float sum = row[0] + row[1] + row[2] + row[3];
        CHECK(approx(sum, 1.0f, 1e-4f), "softmax_row 和 == 1");
        // 最大值位置（10.0）得到最大概率
        CHECK(row[3] > row[0] && row[3] > row[1] && row[3] > row[2], "softmax_row 最大值概率最大");
    }

    // --- silu（Swish）：已知值 ---
    {
        float a[1] = {0.0f};
        kernel::silu(a, 1);
        CHECK(approx(a[0], 0.0f, 1e-6f), "silu(0) == 0");

        // 大正数：silu 渐近于恒等
        float b[1] = {10.0f};
        kernel::silu(b, 1);
        CHECK(approx(b[0], 10.0f, 1e-3f), "silu(10) ≈ 10（大正数趋于恒等）");

        // 大负数：silu 渐近于 0
        float c[1] = {-10.0f};
        kernel::silu(c, 1);
        CHECK(approx(c[0], 0.0f, 1e-3f), "silu(-10) ≈ 0（大负数趋于 0）");
    }

    // --- rms_norm：W=1（一维 [3] F32），常数向量 -> 每元素 ≈ 1 ---
    {
        float x[3] = {2.0f, 2.0f, 2.0f};
        float w[3] = {1.0f, 1.0f, 1.0f};
        float out[3];
        kernel::rms_norm(x, w, 3, 1e-5f, out);
        CHECK(approx(out[0], 1.0f, 1e-3f) && approx(out[1], 1.0f, 1e-3f) &&
              approx(out[2], 1.0f, 1e-3f), "rms_norm 常数向量(w=1) 归一化为 1");
    }

    // --- matmul / dequant_row：本地 F32 权重 [n_in=2, n_out=3] ---
    {
        // R0={1,2}, R1={3,4}, R2={5,6}
        float wdata[6] = {1, 2, 3, 4, 5, 6};
        ggml::ggml_tensor W;
        std::memset(&W, 0, sizeof(W));
        W.type = ggml::GGML_TYPE_F32;
        W.ne[0] = 2;
        W.ne[1] = 3;
        W.data = wdata;

        float x[2] = {1.0f, 1.0f};
        float out[3];
        std::vector<float> scratch(2);
        kernel::matmul(x, &W, 2, 3, out, scratch.data());
        CHECK(approx(out[0], 3.0f) && approx(out[1], 7.0f) && approx(out[2], 11.0f),
              "matmul(F32) out[j]=Σ x[i]*W[j][i]");

        float row0[2];
        kernel::dequant_row(&W, 0, row0); // dequant F32 行 0 应原样
        CHECK(approx(row0[0], 1.0f) && approx(row0[1], 2.0f), "dequant_row(F32) 原样读出");
    }

    if (g_fail == 0)
    {
        std::printf("\nPASS kernel 算子测试\n");
        return 0;
    }
    std::printf("\nFAIL: %d 项断言失败\n", g_fail);
    return 1;
}
