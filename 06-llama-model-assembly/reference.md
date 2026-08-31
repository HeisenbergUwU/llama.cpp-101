# 06 章 · 参考源码对照

本章「模型语义：Layer（层权重语义组装）」主要参考：

- `llama.cpp/src/llama-model.cpp` —— `llama_model_base::load_hparams`（KV -> hparams 的读取、默认值、派生算法）
- `llama.cpp/src/llama-hparams.h/.cpp` —— `llama_hparams` 字段与派生 getter（`n_head`/`n_embd_head_k`/`n_gqa`/`n_embd_k_gqa`/`n_rot`）
- `llama.cpp/src/llama-arch.cpp` —— KV 字符串表（`<arch>.<subkey>` 的键名）+ `LLM_TN_*` tensor 名常量表
- `llama.cpp/src/models/llama.cpp` —— `load_arch_tensors`（plain llama 的装配：tok_embd/output_norm/output，output 缺失复用 token_embd，见 41-46 行）
- `llama.cpp/src/llama-model.h` —— `struct llama_layer`（每层权重 bag 的字段命名）

做法（最小化结构版）：
- 06 章 = 在 05 的「裸张量集合」之上，把 110 个 `ggml_tensor` 按名重组成「有语义的模型」。
- `HParams`：只保留 tinybrainbot（plain llama）需要的 ~10 个字段，不抄上游上百字段。
- `Layer`/`Model`：字段名对齐上游 `llama_layer`；`Model::output` 允许缺失并绑定 `token_embd`（权重绑定 / tied lm_head）。
- **Vocab 不在此章**：已移出，作为后续独立章节（tokenize/detokenize 与 BPE）。
- 仍不建图、不执行、不拷权重——算子是后续章节。
