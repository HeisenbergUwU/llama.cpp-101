# 03 - 迷你 ggml 加载层（池子分配函数）

> 在 02 建好的数据类型之上，把「往池子里实例化张量」的 API 真正写出来。

## 做什么

02 章只定义了 `ggml_type` / `ggml_tensor` / `ggml_context` / `ggml_object` 这些**结构**，但还没有任何能往池子塞东西的函数。本章补上这一层——把 `ggml_init`（建池）、`ggml_new_tensor_*`（实例化张量 + `nb[]` 换算）、`ggml_set_name`、`ggml_nbytes` 等**池子分配 API 真正实现**出来，让「往池子里加载张量」可运行。

只做「加载」，仍不建图、不执行。

## 怎么跑

```bash
cd 03-ggml-build-context && make run
```

## 关键文件

| 文件 | 作用 |
|------|------|
| `include/ggml.h` | `namespace ggml`：类型 + 池子分配函数声明 |
| `src/ggml.cpp` | `ggml_init` / `ggml_new_object` / `ggml_new_tensor_*` / `nb[]` 换算 / `ggml_nbytes` 的实现 |
| `tests/test-ggml-build-context.cpp` | 手写测试：建池、逐个实例化张量、校验 `nb[]`/`nbytes` |

## 对照上游

- `llama.cpp/ggml/src/ggml.c` —— `ggml_init`、`ggml_new_tensor_impl`、`ggml_set_name`、`ggml_nbytes`
- `llama.cpp/ggml/include/ggml.h` —— 上述函数声明与 `ggml_tensor`/`ggml_context` 用法

> 更精简的「参考源码对照」见 `reference.md`。

---

下一章：**04 - 文件 IO 封装层**（`llama_file` + `llama_mmap`，把裸系统调用封装成类型）。
