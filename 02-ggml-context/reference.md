# 02 章 · 参考源码对照

本章「迷你 ggml：`ggml_context` + `ggml_tensor` 数据结构」主要参考：

- `llama.cpp/ggml/src/ggml.c` —— `ggml_init`（建池）、`ggml_new_object`（池子分配）、`ggml_new_tensor_impl`（建张量 + nb[] 换算 + `no_alloc`）、`ggml_set_name`、`ggml_nbytes`
- `llama.cpp/ggml/include/ggml.h` —— `ggml_tensor`/`ggml_context`/`ggml_init_params`/`ggml_type` 声明

做法：只复刻「用一块连续内存装一批带形状的张量结构」。不建图、不执行、不碰文件字节。
真正把 GGUF 元数据变成这些 tensor 并挂权重 → 03 章（对照 `llama_model_loader`/`llama_model`）。
