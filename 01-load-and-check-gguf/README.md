# 01 - 加载并校验 GGUF（load + check）

> 裸 GGUF 解析器：用 `FILE*`/`fread` 顺序读文件，解析出 header、KV pairs、tensor 元数据，并做一致性校验。不建模型、不碰推理。

## 做什么
- 解析 GGUF 文件头（`magic`、`version`、`n_tensors`、`n_kv`）。
- 按规则读 n_kv 个 KV pairs（字符串 / 标量 / ARRAY 三种值的字节排列）。
- 按规则读 n_tensors 个 tensor info（name / n_dims / ne[] / type / offset）。
- 数据对齐：读到 blob 前向上取整到 `alignment`（默认 32）的倍数。
- 一致性校验：offset/size bounds（`offs + nbytes <= file_size`）、张量名不重复、版本号合理。
- 范围限定：只讲 GGUF 文件格式；张量内部量化布局属 ggml 量化格式，不在本章。

> 本机实测 tinybrainbot：magic=`GGUF`、version=3、n_tensors=110、n_kv=32、architecture=`llama`、alignment=32、tensor 类型只用 F32（×25）和 F16（×85）。

## 怎么跑
```bash
cd 01-load-and-check-gguf && make run
```

## 关键文件
| 文件 | 作用 |
|------|------|
| `include/gguf.h` | `gguf_context` / `gguf_tensor_info` / `gguf_kv` 结构定义 |
| `src/gguf.cpp` | 裸 `FILE*`/`fread` 顺序解析 header / KV / tensor info / 对齐 / 校验 |
| `tests/test-load-gguf.cpp` | 手写测试：解析并校验默认模型 `../resources/tinybrainbot-...f16.gguf` |

## 对照上游
- `llama.cpp/ggml/include/gguf.h` —— GGUF 结构定义 + 读取 C API（`gguf_type` 枚举）
- `llama.cpp/ggml/src/gguf.cpp` —— `gguf_init_from_file` 解析实现（KV / tensor / 对齐 / 校验）
- `llama.cpp/src/llama-model-loader.h` —— `llama_tensor_weight` 的 bounds 校验思路

---
下一章：**02 - 迷你 ggml 数据结构层**（`ggml_context` + `ggml_tensor` 的类型与内存池绑定，只建结构不写函数）。
