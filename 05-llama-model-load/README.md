# 05 - 建 llama_model 聚合对象 + 加载权重

> 用 03 的迷你 ggml 把 GGUF 元数据变成 110 个 `ggml_tensor`，用 mmap 零拷贝挂权重。

## 做什么

这是「把权重拿到内存」的关键一步（对比 01 只读元数据）。`load_model` 的 5 步：

1. `llama_file` 打开文件（拿 fd + size）
2. `llama_mmap` 映射整个文件（零拷贝）
3. `gguf::gguf_load` 解析元数据（拿到 `n_tensors`）
4. `ggml_init` 建池（`no_alloc=true`，只装结构，大小按 tensor 数动态算）
5. 循环每个 tensor：`ggml_new_tensor` 实例化 + `set_name` + `data` 指向 mmap 区

`llama_model` 只**持有** 110 个 tensor 对象 + mmap 映射，不建图、不执行、不做 hparams/vocab、不拷权重（`no_alloc=true`，`data` 直接指 mmap 区）。

## 怎么跑

```bash
cd 05-llama-model-load && make run
```

## 关键文件

| 文件 | 作用 |
|------|------|
| `include/llama-model.h` | `struct llama_model`（mmap + ctx + tensors） |
| `src/llama-model.cpp` | `load_model`：5 步流程 + RAII 析构（先 munmap 再 ggml_free） |
| `tests/test-llama-model.cpp` | 手写测试：110 个 tensor 齐全 + mmap 零拷贝 |

## 对照上游

- `llama.cpp/src/llama-model-loader.*` —— `llama_model_loader`、`llama_tensor_weight`（bounds 校验）、`load_data_for`（`data = mapping->addr() + offs` 零拷贝）
- `llama.cpp/src/llama-model.*` —— `llama_model` 聚合对象 + `load_tensors`
- `llama.cpp/src/llama.cpp` —— `llama_model_load`（顶层编排）

> 更精简对照见 `reference.md`。

---

下一章：**06 - 模型语义（HParams + Layer/Model 组装）**（把平铺的裸张量挂成语义结构）。
