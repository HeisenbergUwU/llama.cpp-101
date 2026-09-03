#pragma once

#include <cstdint>

#include "ggml.h" // ggml::ggml_tensor（dequant_row / matmul 反量化权重用）

namespace kernel
{

    // IEEE 754 半精度 -> 单精度（无三方依赖）。普通声明（非 inline）：定义在 kernel.cpp，
    // 供外部链接——加 inline 就必须把函数体放头文件，否则其他编译单元链接找不到符号。
    float fp16_to_fp32(uint16_t h);

    // 把权重张量 W 的第 row 行反量化成 F32 到 dst。整章唯一懂 F16 字节布局的函数：
    //   F16 -> 逐元素 fp16_to_fp32；F32 -> memcpy。dst 长度 = W 的 n_in = ne[0]。
    void dequant_row(const ggml::ggml_tensor *W, int row, float *dst);

    // RMS 归一化（逐行）：out[j] = x[j] / rms * w[j]，rms = sqrt(mean(x^2) + eps)。
    // w 是一维 [n] F32 weight（attn_norm/ffn_norm/output_norm，直接读 data）。
    void rms_norm(const float *x, const float *w, int n, float eps, float *out);

    // 矩阵乘（权重在前）：out[j] = Σ_i x[i]·Wrow[j][i]，j in 0..n_out-1。W 是 [n_in, n_out]
    // （ne[0]=n_in 输入列、ne[1]=n_out 输出行，行主序）。scratch 长 n_in 供反量化当前行复用。
    void matmul(const float *x, const ggml::ggml_tensor *W, int n_in, int n_out, float *out, float *scratch);

    // SiLU（Swish 无参数）：in place，x = x/(1+exp(-x))。
    void silu(float *x, int n);

    // RoPE（NEOX / interleaved）：in place，对前 n_rot 维按对旋转。
    // 每对 (i, i+n_rot/2)，i in 0..n_rot/2-1：x[i]=x0*c - x1*s；x[i+n_rot/2]=x0*s + x1*c。
    void rope_inplace(float *x, int n_rot, int pos, float base);

    // 单行 softmax（in place）：减自身最大值后指数并归一，行内和 = 1。
    void softmax_row(float *x, int n);

    // 带掩码+缩放 softmax（in place，对齐上游 soft_max_f32 掩码/缩放路径）：
    // 先 x[i]=scale·x[i]+(mask?mask[i]:0) 再 softmax_row；n=行内元素数，mask NULL 只缩放不掩码。
    void soft_max_ext(float *x, const float *mask, float scale, int n);

} // namespace kernel