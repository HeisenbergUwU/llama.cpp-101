# 06 章 · 参考源码对照

本章「模型语义：Layer（层权重语义组装）」主要参考：

- `llama.cpp/src/llama-model.cpp` —— `llama_model_base::load_hparams`（KV -> hparams 读取、默认值、派生算法）
- `llama.cpp/src/llama-hparams.h/.cpp` —— `llama_hparams` 字段与派生 getter（`n_head`/`n_embd_head_k`/`n_gqa`/`n_embd_k_gqa`/`n_rot`）
- `llama.cpp/src/llama-arch.cpp` —— KV 字符串表（`<arch>.<subkey>` 键名）+ `LLM_TN_*` tensor 名常量表
- `llama.cpp/src/models/llama.cpp` —— `load_arch_tensors`（plain llama 装配：tok_embd/output_norm/output，output 缺失复用 token_embd，见 41-46 行）
- `llama.cpp/src/llama-model.h` —— `struct llama_layer`（每层权重 bag 字段命名）

做法（最小化结构版）：
- 在 05 的「裸张量集合」上把 110 个 `ggml_tensor` 按名重组成「有语义的模型」。
- `HParams`：只保留 tinybrainbot（plain llama）需要的 ~10 个字段，不抄上游上百字段。
- `Layer`/`Model`：字段名对齐 `llama_layer`；`Model::output` 允许缺失并绑定 `token_embd`（tied lm_head）。
- **Vocab 不在此章**：移出为独立章节（tokenize/detokenize 与 BPE）。
- 仍不建图、不执行、不拷权重——算子是后续章节。
