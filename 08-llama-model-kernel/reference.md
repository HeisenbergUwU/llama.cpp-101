# 08 章 · 参考源码对照

本章「完整前向」主要参考：

- `llama.cpp/src/models/llama.cpp` —— `graph<false>`（解码器主前向：查表 build_inp_embd、两处残差 ADD、`LLM_FFN_SILU` + `LLM_FFN_PAR` 的 SwiGLU 接法、tied lm_head）
- `llama.cpp/src/llama-graph.cpp` —— `build_norm`（RMSNorm）、`build_qkv`（QKV 投影）、`build_attn_mha`（GQA 分组注意力）、`build_ffn`（SwiGLU-PAR）、`build_inp_embd`、`build_causal_mask`
- `llama.cpp/ggml/src/ggml-cpu/ops.cpp` —— `rms_norm_f32`、`soft_max_f32`、`rope_f32` 内核
- `llama.cpp/src/llama-vocab.h` —— `tokenize`（SPM，本测试取 token 序列用）

做法（最小化结构版）：
- Scheme A 直接循环，不建计算图；权重留 mmap，按行反量化进复用 scratch（option c）。
- 三个接缝（KVSlice / build_causal_mask / forward）为 11 章 KV cache 预留。
