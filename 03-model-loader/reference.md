# 03 章 · 参考源码对照

本章「建 `llama_model` 并加载权重」主要参考：

- `llama.cpp/src/llama-model-loader.cpp/.h` —— `llama_model_loader` 构造、`get_weight`/`require_weight`、`create_tensor`、`init_mappings`（mmap）、`load_data_for`（零拷贝挂数据）
- `llama.cpp/src/llama-model.cpp/.h` —— `llama_model` 聚合对象、`llama_model_create`、`load_tensors`
- `llama.cpp/src/llama.cpp` —— `llama_model_load`（顶层编排：loader → model_create → load_tensors）

教程 `llama_model::load` 就是这套编排的裁剪版（跳过 hparams/vocab，只到「持有 110 个 tensor 对象 + mmap 零拷贝」）。
建图/执行、GPU 布局、多分片等后续才做。
