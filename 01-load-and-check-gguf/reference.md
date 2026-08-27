# 01 章 · 参考源码对照

本章「加载并校验 GGUF」主要参考：

- `llama.cpp/ggml/src/gguf.cpp` —— `gguf_init_from_file` → `_file_ptr` → `_from_reader`（读取流程）；`gguf_context`/`gguf_tensor_info`/`gguf_kv` 结构定义
- `llama.cpp/ggml/include/gguf.h` —— `gguf_context`/`gguf_tensor_info`/`gguf_init_params` 声明
- `llama.cpp/src/llama-model-loader.h` —— `llama_tensor_weight` 的 bounds 校验思路
