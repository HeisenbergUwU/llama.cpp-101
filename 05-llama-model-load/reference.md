# 04 章 · 参考源码对照

本章「建 `llama_model` 聚合对象并加载权重」主要参考：

- `llama.cpp/src/llama-mmap.cpp/.h` —— `llama_mmap`（`mmap` 整个文件、`addr()`/`size()`、POSIX `mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0)`）
- `llama.cpp/src/llama-model-loader.cpp/.h` —— `llama_model_loader` 构造、`llama_tensor_weight`（`offs = gguf_get_data_offset + gguf_get_tensor_offset` 的 bounds 校验）、`get_tensor_meta`、`load_data_for`（`cur->data = mapping->addr() + offs` 零拷贝挂数据）、`init_mappings`
- `llama.cpp/src/llama-model.cpp/.h` —— `llama_model` 聚合对象（持有 `ctx` / tensor 容器 / mmap）、`load_tensors`
- `llama.cpp/src/llama.cpp` —— `llama_model_load`（顶层编排：loader → model_create → load_tensors）

做法（最小化结构版）：
- 04 章 = **用 03 的迷你 ggml** 把 01 的 GGUF 元数据变成 110 个 `ggml_tensor`，并用 **mmap 零拷贝**挂权重。
- `llama_model` 只做到「**持有** 110 个 tensor 对象 + mmap 映射」这一层：**不建图、不执行、不做 hparams/vocab、不拷权重**（`no_alloc=true`，`data` 直接指 mmap 区）。
- 归属：`gguf_context`(01) 加载时读 offset、用完即弃；`ggml_context`(03) 管 tensor 结构池；mmap 映射管数据。三者由 `llama_model` 撮合，析构顺序 = 先 `munmap` 再 `ggml_free`。
- 建图/执行、GPU 布局、多分片等后续章节才做。
