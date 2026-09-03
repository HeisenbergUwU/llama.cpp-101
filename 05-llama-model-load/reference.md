# 05 章 · 参考源码对照

本章「建 `llama_model` 聚合对象并加载权重」主要参考：

- `llama.cpp/src/llama-mmap.cpp/.h` —— `llama_mmap`（POSIX `mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0)`、`addr()`/`size()`）
- `llama.cpp/src/llama-model-loader.cpp/.h` —— `llama_model_loader`、`llama_tensor_weight`（offset bounds 校验）、`load_data_for`（`cur->data = mapping->addr() + offs` 零拷贝挂数据）、`init_mappings`
- `llama.cpp/src/llama-model.cpp/.h` —— `llama_model` 聚合对象 + `load_tensors`
- `llama.cpp/src/llama.cpp` —— `llama_model_load`（顶层编排：loader → model_create → load_tensors）

做法（最小化结构版）：
- 用 03 的迷你 ggml 把 01 的 GGUF 元数据变成 110 个 `ggml_tensor`，mmap 零拷贝挂权重。
- `llama_model` 只「持有」tensor + mmap：不建图、不执行、不做 hparams/vocab、不拷权重（`no_alloc=true`）。
- 归属：`gguf_context`(01) 加载时读 offset、用完即弃；`ggml_context`(03) 管 tensor 结构池；mmap 管数据。析构顺序 = 先 `munmap` 再 `ggml_free`。
- 建图/执行、GPU 布局、多分片等后续章节才做。
