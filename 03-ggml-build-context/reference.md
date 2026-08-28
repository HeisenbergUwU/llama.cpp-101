# 03 章 · 参考源码对照

本章「迷你 ggml 的函数 / 加载层：把 02 的数据结构变成能真正往池子里实例化张量的 API」主要参考：

- `llama.cpp/ggml/src/ggml.c` —— `ggml_init`（建池）、`ggml_new_object`（池子切块分配）、`ggml_new_tensor_impl`（实例化张量 + nb[] 换算 + no_alloc/view）、`ggml_set_name`、`ggml_nbytes`
- `llama.cpp/ggml/include/ggml.h` —— 上述函数的声明与 `ggml_tensor`/`ggml_context` 用法

做法：02 只建好了结构；03 把 `ggml_init`/`ggml_new_tensor_*` 等**池子分配 API 真正写出来**，让「往池子里加载张量」可运行。只做加载，仍不建图、不执行。
真正把 GGUF 元数据读成 110 个 tensor 并 mmap 挂权重 → 04 章（对照 `llama_model_loader`/`llama_model`）。
