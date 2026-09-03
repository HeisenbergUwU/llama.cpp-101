# 08 - 算子内核（kernel 层）

> 独立 `namespace kernel` 的纯计算算子：反量化、RMSNorm、matmul、SiLU、RoPE、softmax，供后续完整前向复用。

## 做什么
- 7 个无状态算子，输入输出均走裸 `float*`（带权重者走 `ggml::ggml_tensor*`）：
  `fp16_to_fp32` / `dequant_row` / `rms_norm` / `matmul` / `silu` / `rope_inplace` / `softmax_row`
- `dequant_row`：整章唯一懂 F16 字节布局的函数（F16 → `fp16_to_fp32`；F32 → `memcpy`），按行反量化进复用 scratch
- `fp16_to_fp32`：手写 IEEE 754 半精度 → 单精度（含 0/非规格化/无穷/NaN），无三方依赖
- `ROPE` 为 NEOX / interleaved（in place，只转前 `n_rot` 维）
- 与前向/模型语义解耦（不依赖 forward/KVSlice），可单独编译测试
- 为 13+ 章「逐算子原位深入」预留落点：改一个算子只需改 kernel.cpp + test-kernel.cpp

## 怎么跑（CMake，kernel 独立构建）
```bash
cd 08-llama-model-kernel
cmake -B build-kernel -S src/kernel && cmake --build build-kernel
./build-kernel/test-kernel   # 只测 7 个算子，无需模型权重
```

## 关键文件
| 文件 | 作用 |
|------|------|
| `include/kernel/kernel.h` | `namespace kernel`：7 个算子声明（只依赖 ggml） |
| `src/kernel/cpu/kernel.cpp` | 算子 CPU 实现 |
| `src/kernel/CMakeLists.txt` | kernel 子项目：kernel 静态库 + `test-kernel` + `run-kernel` 便捷目标 |
| `tests/test-kernel.cpp` | kernel 单独测试（无需模型，本地 F32 tensor 喂 matmul/dequant_row） |

## 对照上游
- `llama.cpp/ggml/src/ggml-cpu/ops.cpp` —— `ggml_rms_norm_f32`、`ggml_soft_max_f32`、`ggml_rope_f32` 内核
- `llama.cpp/ggml/src/ggml.h` —— `ggml_fp16_to_fp32`、量化 block 结构体（F16/F32 布局）
- `llama.cpp/src/models/llama.cpp` —— `graph<false>` 里对 RMSNorm/线性/FFN 算子的搭建方式（后续前向章参照）

---
下一章：**09 - 自回归采样**（拿到 logits 后 greedy 逐 token 生成，处理 EOG/max_tokens；进度见 `ROADMAP.md`）。
