# 00 - llama.cpp 是什么

> 用 C/C++ 写的 LLM 推理引擎，本项目按它的真实分层自写最小版本来讲清底层原理。

## 做什么
- 介绍 llama.cpp：从在 MacBook 上跑 LLaMA 起家，现覆盖 CPU 与 GPU（CUDA / Metal / Vulkan）。
- 拆解两大核心问题：权重怎么存读（GGUF，01 章）与推理怎么算（ggml 计算图）。
- 梳理仓库三层架构：应用层（examples / server）、模型层（llama-model-loader 读 GGUF）、计算层（ggml 张量运算）。
- 列出 `llama.cpp/` 关键目录分布（`include`/`src`/`ggml`/`examples`/`common`/`tests`）。
- 对照一次推理的完整流程，映射到后续各章分工。
- 说明教学路线：不教"用 llama.cpp"，而是每层自写最小实现再对照上游源码。

## 怎么跑

本章只有讲解，无代码、无 Makefile，不涉及运行。

## 关键文件

| 文件 | 作用 |
|------|------|
| `README.md` | 本章全部讲解内容（无 src / include / tests） |

## 对照上游

- `llama.cpp/include/llama.h` —— 对外 C API（加载、context、分词、采样）
- `llama.cpp/src/` —— 模型加载（`llama-model-loader.*`）、推理调度、采样、KV 缓存
- `llama.cpp/ggml/` —— ggml 张量库：计算图、张量运算、CPU 内核、量化格式
- `llama.cpp/ggml/include/gguf.h` —— GGUF 文件格式定义 + 读写 API（01 章核心对象）
- `llama.cpp/examples/` —— 可执行 demo：`main`、`server`、`simple`、`embedding`

---
下一章：**01 - 加载并校验 GGUF**（裸解析器只读 GGUF，读 header / KV / tensor 元数据并一致性校验，先不碰推理）。
