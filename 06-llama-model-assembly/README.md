# 06 - 模型语义（HParams + Layer/Model 组装）

> 在 05 的「裸张量集合」之上，把 110 个 `ggml_tensor` 按名重组成「有语义的模型」。

## 做什么

05 拿到的是一堆平铺的 `ggml_tensor`（按文件名乱序，如 `blk.0.attn_q.weight`）。本章把它们**挂成语义结构**：

- **`HParams` + `parse_hparams`**：从 GGUF KV 解析超参数（`n_embd=768`、`n_layer=12`、GQA `n_head=12/n_head_kv=4`、SwiGLU 等）+ 派生 getter（`n_gqa()`、`n_embd_k_gqa()`）。
- **`Layer` / `Model`**：字段名对齐上游 `llama_layer`。每层一个 bag（`attn_norm`/`wq`/`wk`/`wv`/`wo`/`ffn_norm`/`gate`/`up`/`down` 共 9 个权重）+ 根张量（`token_embd`/`output_norm`/`output`）。`output` 缺失时绑定 `token_embd`（权重绑定 / tied lm_head）。
- **`assemble_model`**（由 `build_semantics` 更名）：按 tensor 名把 `llm.tensors` 重组进 `Model`。

仍不建图、不执行、不拷权重——算子是后续章节。**Vocab 不在此章**（已移出为 07 章）。

## 怎么跑

```bash
cd 06-llama-model-assembly && make run
```

## 关键文件

| 文件 | 作用 |
|------|------|
| `include/llama-hparams.h` / `src/llama-hparams.cpp` | `HParams` + `parse_hparams` + 派生 getter |
| `include/llama-model.h` | `Layer` / `Model` / `llama_model` + `assemble_model` 声明 |
| `src/llama-model.cpp` | `assemble_model` 实现（按名查找并挂接） |
| `tests/test-llama-model-assembly.cpp` | 手写测试：hparams 实测值 + 层数/张量完备性/形状 |
| `model-graph.md` | 模型结构图（Mermaid + ASCII：架构 / 类图 / 端到端流程 / GQA / FFN） |

## 对照上游

- `llama.cpp/src/llama-hparams.h/.cpp` —— `llama_hparams` 字段与派生 getter
- `llama.cpp/src/llama-model.h` —— `struct llama_layer`（每层权重 bag）
- `llama.cpp/src/models/llama.cpp` —— `load_arch_tensors`（plain llama 装配，output 缺失复用 token_embd）

> 更精简对照见 `reference.md`。

---

下一章：**07 - 词表 Vocab（tokenize / detokenize）**（从 `tokenizer.ggml.*` KV 建词表）。
