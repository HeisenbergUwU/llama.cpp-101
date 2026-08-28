# 02 章 · 参考源码对照

本章「迷你 ggml：`ggml_context` + `ggml_tensor` 数据结构」主要参考：

- `llama.cpp/ggml/include/ggml.h` —— `ggml_tensor`/`ggml_context`/`ggml_init_params`/`ggml_type`/`ggml_object_type` 声明
- `llama.cpp/ggml/src/ggml.c` —— `ggml_object`/`ggml_context` **内部布局**（`offs`/`size`/`next`、`objects_begin/end`、`mem_buffer_owned`、`no_alloc`）

做法：只复刻「用一块连续内存（池子）装一批带形状的张量结构」，建好类型与布局，**不写任何函数**。
池子分配的加载函数（`ggml_init`/`ggml_new_object`/`ggml_new_tensor_*`/`ggml_set_name`/`ggml_nbytes`）→ 03 章；
把 GGUF 元数据变成 110 个 tensor 并 mmap 挂权重 → 04 章（对照 `llama_model_loader`/`llama_model`）。
