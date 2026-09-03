// kernel/cpu/kernel.cpp - 08 章「算子内核」CPU 实现。纯计算算子（Scheme A 不建图）与 forward/KVSlice 解耦，可单独编译+测试。
// 权重在 mmap 上，option(c) 按行反量化进复用 scratch（F32 累加避免另造副本），唯一懂 F16 布局的是 dequant_row，13 章+深挖只改这里。

#include "kernel/kernel.h"

#include <cmath>
#include <cstring>

namespace kernel
{

    // ---- IEEE 754 半精度->单精度：h 布局 s(1)e(5)m(10)，常规值=(-1)^s×2^(e-15)×1.m ----
    // 不能标 inline（头文件也是普通声明），否则函数不导出外部符号，test-kernel.cpp 单独链接会 undefined symbol。
    float fp16_to_fp32(uint16_t h)
    {
        const uint16_t s = (h >> 15) & 0x1;
        const uint16_t e = (h >> 10) & 0x1F;
        const uint16_t m = h & 0x3FF;

        if (e == 0)
        {
            // 零 / 非规格化：(-1)^s × 2^-14 × (m / 1024)
            if (m == 0)
            {
                return s ? -0.0f : 0.0f; // 正/负零
            }
            // 非规格化：尾数无隐式 1，值 = m × 2^-24（= 2^-14 × 2^-10）
            return (s ? -1.0f : 1.0f) * (float)m * 0x1p-24f;
        }
        if (e == 0x1F)
        {
            // 无穷 / NaN：m==0 -> ±inf，否则 NaN
            return (m == 0) ? (s ? -INFINITY : INFINITY) : NAN;
        }

        // 常规：指数字段已带偏置 15，换算成单精度带偏置 127 的指数 = e + 112
        const uint32_t f32 = ((uint32_t)s << 31) | ((uint32_t)(e + 112) << 23) | ((uint32_t)m << 13);
        float out;
        std::memcpy(&out, &f32, sizeof(out));
        return out;
    }

    // ---- 反量化权重的一行：W->data 指向 mmap 原始字节，第 row 行起点=row×ne[0]×每元素字节数 ----
    // F16 逐元素 fp16_to_fp32；F32 直接 memcpy（4 字节/元素且小端一致）。
    void dequant_row(const ggml::ggml_tensor *W, int row, float *dst)
    {
        const int64_t n_in = W->ne[0]; // 每行元素数
        if (W->type == ggml::GGML_TYPE_F32)
        {
            const float *src = (const float *)W->data + (int64_t)row * n_in;
            std::memcpy(dst, src, (size_t)n_in * sizeof(float));
        }
        else // GGML_TYPE_F16
        {
            const uint16_t *src = (const uint16_t *)W->data + (int64_t)row * n_in;
            for (int64_t i = 0; i < n_in; ++i)
            {
                dst[i] = fp16_to_fp32(src[i]);
            }
        }
    }

    // ---- RMS 归一化（逐行） ----
    // rms = sqrt(mean(x^2)+eps)；out[j] = x[j]/rms*w[j]；w 是一维 [n] F32 weight（直接读 data）。
    void rms_norm(const float *x, const float *w, int n, float eps, float *out)
    {
        float sum = 0.0f;
        for (int j = 0; j < n; ++j)
        {
            sum += x[j] * x[j];
        }
        const float rms = std::sqrt(sum / (float)n + eps);
        const float inv = 1.0f / rms;
        for (int j = 0; j < n; ++j)
        {
            out[j] = (x[j] * inv) * (w[j]);
        }
    }

    // ---- 矩阵乘（权重在前）：W 是 [n_in,n_out]（ne[0]=列输入、ne[1]=行输出，行主序）----
    // out[j]=Σ_i x[i]*Wrow[j][i]，Wrow[j] 反量化到 scratch（长度 n_in）。
    void matmul(const float *x, const ggml::ggml_tensor *W, int n_in, int n_out, float *out, float *scratch)
    {
        for (int j = 0; j < n_out; ++j)
        {
            dequant_row(W, j, scratch);
            float acc = 0.0f;
            for (int i = 0; i < n_in; ++i)
            {
                acc += x[i] * scratch[i];
            }
            out[j] = acc;
        }
    }

    // ---- SiLU（Swish 无参数）：in place ----
    void silu(float *x, int n)
    {
        for (int i = 0; i < n; ++i)
        {
            x[i] = x[i] / (1.0f + std::exp(-x[i]));
        }
    }

    // ---- RoPE（NEOX / interleaved）：in place ----
    // 只处理前 n_rot 维；第 i 与第 i+n_rot/2 配对。x 视为"一行 n_rot 长"。
    void rope_inplace(float *x, int n_rot, int pos, float base)
    {
        const int half = n_rot / 2;
        for (int i = 0; i < half; ++i)
        {
            const float theta = (float)pos * std::pow(base, -2.0f * (float)i / (float)n_rot);
            const float c = std::cos(theta);
            const float s = std::sin(theta);
            const float x0 = x[i];
            const float x1 = x[i + half];
            x[i] = x0 * c - x1 * s;
            x[i + half] = x0 * s + x1 * c;
        }
    }

    // ---- 单行 softmax（in place）：减最大值后指数并归一 ----
    void softmax_row(float *x, int n)
    {
        float mx = x[0];
        for (int i = 1; i < n; ++i)
        {
            if (x[i] > mx)
            {
                mx = x[i];
            }
        }
        float sum = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            x[i] = std::exp(x[i] - mx);
            sum += x[i];
        }
        const float inv = 1.0f / sum;
        for (int i = 0; i < n; ++i)
        {
            x[i] *= inv;
        }
    }

} // namespace kernel
