# 11 - 自回归采样 + 跑起来：一个文字一个文字地推理

> 在 10 章整图之上补最后一环——**采样器 + 解码循环**：喂 prompt，每步重建一张图、全量重算（无 KV cache），对末 token 的 logits 行 greedy argmax 采出下一个 token，直到 EOG 或 max_tokens，吐出人能读的文本。

## 做什么
- 新增 `llama::Sampler`：`sample(logits, n_vocab) -> id`，在一行 logits 里挑最大的 token（greedy），签名留好便于后续换 top-k/top-p。
- `llama_model::generate()`（内联在 llama-model.h）：自回归解码循环——tokenize prompt → 循环重建整图 → `compute` 得 logits → greedy 采样 → `is_eog` 判断 → detokenize 打印，直到 EOG 或 max_tokens。
- 无 KV cache 取舍：每步用完整历史重建一张与 10 章一致的整图，逻辑最简、易讲清解码闭环；代价 O(N²)，8GB 机器跑 ≤20 token 的教程例子没问题。
- 为「每步重建图」健壮补三处：池子按 N 动态估计（`estimate_pool`）、`build_model_graph` 逐点判空（`need()`）、新增 `ggml_mem_used()`（对齐上游 `ggml_used_mem`）。
- 实测：默认提示 `"The capital of France is"` 生成 ` Paris.`，命中 EOG 早停；另有 `llama-cli` 命令行应用。

## 怎么跑
```bash
cd 11-llama-model-run-it!
cmake -B build -S . && cmake --build build
./build/test-llama-sample ../resources/tinybrainbot-100m-v3-instruct/tinybrainbot-100m-v3-instruct-f16.gguf
./build/llama-cli ../resources/tinybrainbot-100m-v3-instruct/tinybrainbot-100m-v3-instruct-f16.gguf -n 64
```

## 关键文件
| 文件 | 作用 |
|------|------|
| `include/llama-model.h` | 核心：`generate()` 解码循环内联写在这（挂 `llama_model`）+ `Sampler`/`Generation` 类型 |
| `src/llama-sampler.cpp` | `Sampler::sample`（greedy argmax）实现 |
| `tests/test-llama-sample.cpp` | 可跑测试：加载模型 → `llm.generate(...)` 生成文本 → 校验停止条件 |
| `src/llama-cli.cpp` | 命令行应用：加载模型 → 生成文本 |
| `src/llama-graph.cpp` | 10 章拷贝；本章加 `need()` 逐点判空 |

## 对照上游
- `llama.cpp/src/llama-sampler.h/.cpp` —— `llama_sampler_init_greedy`（`apply` 挑最大 token）、`llama_token_data`/`llama_sampler_sample`
- `llama.cpp/src/llama.cpp` —— `llama_decode`（喂 token 拿 logits）、`llama_generate`（解码循环：decode → sample → 判 `is_eog` → 续/停）
- `llama.cpp/src/llama-vocab.h/.cpp` —— `is_eog`（句末判断，本迷你版 `Vocab::is_eog`）
- `llama.cpp/ggml/include/ggml.h` / `ggml/src/ggml.c` —— `ggml_used_mem`（池子已用字节）

---
下一章待 ROADMAP.md 更新（KV cache / batch 优化）。
