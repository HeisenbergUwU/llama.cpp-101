# 09 - ggml 计算图设施

> 把 08 章直写循环的前向升级为「计算图」表示：建节点、登记、按拓扑执行，为 10 章把整段模型前向建成一张图打基础。

## 做什么
- 实现计算图骨架：`ggml_build_forward_expand` 登记节点、`ggml_graph_compute` 按拓扑序执行。
- `ggml-ops` 提供算子节点构造函数（`ggml_mul_mat`/`ggml_rms_norm`/`ggml_get_rows`…），只建节点不计算。
- `ggml-kernel` 是「图 ↔ 算子」的分发层：把 op 值 switch 到 08 章 kernel 的裸 `float*` 单算子。
- 分层构建：kernel（纯算子，可单独建）→ ggml/ggml-ops → ggml-graph，各自独立静态库。
- 单测无需模型权重，直接验算子与图设施（build/compute）。

## 怎么跑
```bash
cd 09-llm-graph
cmake -B build -S . && cmake --build build
./build/test-ggml-graph
```

## 关键文件
| 文件 | 作用 |
|------|------|
| `src/ggml-graph.cpp` / `include/ggml-graph.h` | 计算图骨架：登记 + 按拓扑执行 |
| `src/ggml-ops.cpp` / `include/ggml-ops.h` | 算子节点构造函数（建节点） |
| `src/ggml-kernel.cpp` / `include/ggml-kernel.h` | 图 ↔ 算子的分发/组合层 |
| `src/kernel/cpu/kernel.cpp` / `include/kernel/kernel.h` | 08 的裸 `float*` 单算子 |
| `tests/test-ggml-graph.cpp` | 无模型单测：算子/图设施 |

## 对照上游
- `llama.cpp/ggml/src/ggml.c` —— `ggml_build_forward_expand`、`ggml_graph_compute`（图骨架）
- `llama.cpp/ggml/src/ggml.c` —— `ggml_mul_mat`/`ggml_rms_norm` 等算子节点构造函数
- `llama.cpp/ggml/src/ggml-cpu/*.c` —— 各 op 的实际内核（分发落点）
- `llama.cpp/ggml/src/ggml-cpu/ops.cpp` —— `rms_norm_f32`/`soft_max_f32`/`rope_f32` 等数值

---
下一章：**10 - llama-model-graph**，把 llama 完整前向建成一张整图。
