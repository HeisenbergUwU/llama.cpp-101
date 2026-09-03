# 10 - 整体模型图：把 llama 完整前向建成一张 ggml 计算图

> 用 09 章的计算图设施，把 08/06 建好的「有语义的模型」整段前向（embed → 12 层 attention+FFN → tied lm_head）建成**一张** `ggml_cgraph`，一次 `compute` 跑出 `logits[N × n_vocab]`。

## 做什么
- `build_model_graph(ctx, model, hp, tokens, N, positions)` 返回 `BuiltGraph{cgraph, logits}`，权重（mmap 叶）作叶子、整图按拓扑执行。
- 用算子节点构造函数登记前向：`get_rows`（embed 查表）→ 逐层 `rms_norm` → QKV `mul_mat` → `rope` → 逐 kv 头注意力 → `mul_mat(wo)`/FFN 残差 → `output_norm` → tied `mul_mat` lm_head。
- GQA 注意力在图上用「逐头 2D」组合表达：`head_extract`（strided 切列）+ `mul_mat` + `soft_max_ext`（掩码+1/√d）+ `concat`（12 头重组）。
- 无 KV cache，全量重算（`n_kv = n_tokens = N`），因果由 `soft_max_ext` 的 mask（`s>t→-inf`）管。
- 为跑 N>1 整图补底层：`mul_mat`/`rms_norm`/`rope` 按行循环、`get_rows` 用 `dequant_row` 反量化 F16、新增 `head_extract`/`concat`（不改数值算法）。
- 实测与 08 直写前向数值对齐：提示 `"The capital of France is"` → `argmax id=7831 ' Paris'`。

## 怎么跑
```bash
cd 10-llama-model-graph
cmake -B build -S . && cmake --build build
./build/test-ggml-graph          # 无模型单测：算子/图设施
./build/test-llama-graph-model ../resources/tinybrainbot-100m-v3-instruct/tinybrainbot-100m-v3-instruct-f16.gguf
```

## 关键文件
| 文件 | 作用 |
|------|------|
| `src/llama-graph.cpp` / `include/llama-graph.h` | 核心：`build_model_graph` 把整段前向建成一张图 |
| `src/ggml-ops.cpp` / `include/ggml-ops.h` | 算子节点构造函数（新增 `head_extract`/`concat`，改造 `rope`/`mul_mat`） |
| `src/ggml-kernel.cpp` | 分发：按行循环修 mul_mat/rms_norm/get_rows/rope + 两个新 op 的 compute |
| `tests/test-llama-graph-model.cpp` | 整体模型图测试（加载真实模型，验 logits 与 08 对齐） |
| `tests/test-ggml-graph.cpp` | 无需模型的算子/图设施单测 |

## 对照上游
- `llama.cpp/src/models/llama.cpp` —— `graph<false>`（标准 llama 解码器主前向）
- `llama.cpp/src/llama-graph.cpp` —— `build_inp_embd`/`build_norm`/`build_qkv`/`build_attn_mha`/`build_ffn`/`build_lora_mm`
- `llama.cpp/ggml/src/ggml-cpu/*.c` —— `soft_max_f32`/`rms_norm_f32`/`rope_f32` 等内核数值

---
下一章：**11 - llama-model-run-it!**，加采样器 + 解码循环，真正跑起来出文本。
