# llama.cpp-101

**简体中文 · [English](#english)**

> 从零打造一个**小而精的 llama.cpp 推理引擎**：一步步拆解，最后得到一个**带服务 API 的可运行项目**。

---

## 💡 为什么创建这个项目？ Why this project?

开始读 llama.cpp 源码时，我发现它**代码量过大**——几十种量化、多后端（CPU/CUDA/Metal/Vulkan）、各种加速路径，生产级 C++ 里塞满了 mmap、多分片、端序处理等大量"必要的复杂度"。

很容易**迷失在代码细节的迷宫里**：不知道从哪读起，也不知道哪些是骨架、哪些是皮毛。

所以我创建这个项目，换一条路：

> **按 llama.cpp 的真实分层，自己动手，一步一步搭一个小而精的推理引擎。** 每一层都对照着读上游源码，把"为什么这么设计"讲清楚；最后汇聚成一个**具有服务 API 功能**的完整项目——既能跑，又看得懂。

---

## 🎯 目标 Goals

| | |
|---|---|
| **小而精** | 刻意保持精简，只保留能讲清原理的最小实现，不堆功能 |
| **一步步** | 严格分阶段渐进，每阶段只做当前阶段的事，不越界 |
| **有产出** | 最终交付一个带服务 API 的推理项目，而不只是一堆讲解 |

---

## 🗺️ 路线图 Roadmap

按 llama.cpp 的真实分层，从 "读文件" 到 "跑推理、再到对外提供 API" 逐章拆解。**最新分章粒度以 `ROADMAP.md` 为准**，本表给出当前进度：

| 章 | 主题 | 对应上游 | 状态 |
|----|------|----------|:----:|
| 00 | 什么是 llama.cpp | — | ✅ |
| 01 | 加载并校验 GGUF（裸解析器） | `gguf.cpp::gguf_init_from_file` | ✅ |
| 02 | 迷你 ggml 数据结构层（类型与布局） | `ggml/src/ggml.h/.c` | ✅ |
| 03 | 迷你 ggml 加载层（池子分配函数） | `ggml/src/ggml.c` | ✅ |
| 04 | 文件 IO 封装层（`llama_file` + `llama_mmap`） | `llama.cpp/src/llama-mmap.*` | ✅ |
| 05 | `llama_model` 聚合对象 + 加载权重（mmap 零拷贝） | `llama-model-loader.*` + `llama-model.*` | ✅ |
| 06+ | 模型语义 / 前向 / 采样 / server | ggml 计算图 + `llama_context` + server | ⬜ 规划中 |

---

## 📚 仓库结构 Layout

```
llama.cpp-101/
├── llama.cpp/                     # 上游 llama.cpp 完整 clone（被研究对象，git 忽略）
├── resources/                     # 测试用模型权重（tinybrainbot 等，git 忽略）
├── 00-what-is-llama-cpp/          # 00 章：llama.cpp 是什么
├── 01-load-and-check-gguf/        # 01 章：加载并校验 GGUF（裸 GGUF 解析器）
├── 02-ggml-context/               # 02 章：迷你 ggml 数据结构层
├── 03-ggml-build-context/         # 03 章：迷你 ggml 加载层（池子分配函数）
├── 04-aggregate-functions/        # 04 章：文件 IO 封装层（llama_file + llama_mmap）
├── 05-llama-model-load/           # 05 章：建 llama_model 聚合对象 + 加载权重
├── ROADMAP.md                     # 权威分章路线图（改动分章前先读它）
└── AGENTS.md                      # 协作者 / 贡献者须知
```

> `llama.cpp/`、`resources/`、`.omo/`、`playground/` 均被 git 忽略；每个可构建章节自带源码、测试与 Makefile（04/05 自包含，含前几章代码的本地拷贝）。

**快速开始：**

```bash
# 01 章：裸 GGUF 解析器，解析并校验 tinybrainbot
cd 01-load-and-check-gguf && make run

# 05 章：完整加载 llama_model（110 个 tensor，mmap 零拷贝），8GB 内存也能跑
cd 05-llama-model-load && make run
```

---

## 🛠️ 方法 Methods

1. **从零实现，不抄上游**：每一章自己写最小实现，刻意保持精简、只关注该章那一个原理。
2. **对照真实源码**：每个实现都标注对应 `llama.cpp/` 内的实际源码位置（相对路径），以真实代码为准，不凭记忆。
3. **轻量可跑**：用 tinybrainbot（约 200MB，F16）当测试模型；测试不引入第三方框架。
4. **最终有产出**：不是停留在讲解，而是汇聚成一个带服务 API 的推理项目。

> 比起"会跑"，我们更在意"讲得清楚原理"——这是整个项目取舍的准则。

---

<a name="english"></a>

## 🌐 English

**llama.cpp-101** — build a **small but precise llama.cpp inference engine** step by step, ending with a **runnable project that exposes a service API**.

### Why this project?

When I started reading the llama.cpp source, I found it **too large** — dozens of quantizations, multiple backends (CPU/CUDA/Metal/Vulkan), and production-grade C++ packed with mmap, multi-shard, endianness handling, and lots of "necessary complexity."

It's all too easy to **get lost in the maze of code details**: not knowing where to start, or which parts are the skeleton and which are the fluff.

So I created this project and went a different way:

> **Follow llama.cpp's real layering and build, by hand, a small, precise inference engine step by step.** At each layer, cross-check against the upstream source to explain *why* it's designed that way — converging into a **full project with a service API** that both runs and is understandable.

### Goals

| | |
|---|---|
| **Small & precise** | Deliberately minimal — keep only what explains the principle, no feature-stuffing |
| **Step by step** | Strictly staged; each phase does only its own work |
| **A deliverable** | Ends in a service-API inference project, not just prose |

### Roadmap

Following llama.cpp's real layering, chapter by chapter from "reading the file" to "inference, then a service API". **Authoritative granularity lives in `ROADMAP.md`**; this table shows current progress:

| Ch | Topic | Upstream reference | Status |
|----|-------|--------------------|:------:|
| 00 | What is llama.cpp | — | ✅ |
| 01 | Load & validate GGUF (bare parser) | `gguf.cpp::gguf_init_from_file` | ✅ |
| 02 | Mini ggml data-structure layer | `ggml/src/ggml.h/.c` | ✅ |
| 03 | Mini ggml load layer (pool allocators) | `ggml/src/ggml.c` | ✅ |
| 04 | File I/O wrapper (`llama_file` + `llama_mmap`) | `llama.cpp/src/llama-mmap.*` | ✅ |
| 05 | `llama_model` aggregate + load weights (mmap zero-copy) | `llama-model-loader.*` + `llama-model.*` | ✅ |
| 06+ | Model semantics / forward / sampling / server | ggml graph + `llama_context` + server | ⬜ planned |

### Layout

```
llama.cpp-101/
├── llama.cpp/                     # upstream clone (studied, git-ignored)
├── resources/                     # test weights (git-ignored)
├── 00-what-is-llama-cpp/          # Ch.00: what is llama.cpp
├── 01-load-and-check-gguf/        # Ch.01: load & validate GGUF (bare parser)
├── 02-ggml-context/               # Ch.02: mini ggml data-structure layer
├── 03-ggml-build-context/         # Ch.03: mini ggml load layer (pool allocators)
├── 04-aggregate-functions/        # Ch.04: file I/O wrapper (llama_file + llama_mmap)
├── 05-llama-model-load/           # Ch.05: llama_model aggregate + load weights
├── ROADMAP.md                     # authoritative chapter roadmap
└── AGENTS.md                      # contributor notes
```

> `llama.cpp/`, `resources/`, `.omo/`, `playground/` are git-ignored; each buildable chapter ships its own source, test, and Makefile (04/05 are self-contained with local copies of prior chapters).

**Get started:**

```bash
# Ch.01: bare GGUF parser, parse & validate tinybrainbot
cd 01-load-and-check-gguf && make run

# Ch.05: full llama_model load (110 tensors, mmap zero-copy), runs on 8GB RAM
cd 05-llama-model-load && make run
```
