# llama.cpp-101

**简体中文 · [English](#english)**

从零讲透 [llama.cpp](https://github.com/ggerganov/llama.cpp) 底层原理的中文教程项目。不教你怎么"用" llama.cpp，而是按它的真实分层，自己动手写简化实现，一步步读懂 GGUF、模型加载与推理。

*English: a hands-on tutorial that explains llama.cpp's internals from scratch — not how to *use* it, but how it actually works, by re-implementing the layers by hand.*

---

## 为什么建这个项目？ Why this project?

llama.cpp 功能极其庞大：几十种量化、多后端（CPU/CUDA/Metal/Vulkan）、各种加速路径。直接打开它的源码，很容易淹没在浩如烟海的细节里——不知道从哪读起，也不知道哪些是骨架、哪些是皮毛。

这个项目的出发点很简单：

- **直接读源码门槛太高。** llama.cpp 是生产级 C++，充斥着 mmap、多分片、端序、量化内核等大量"必要的复杂度"，对初学者几乎是不可读的。
- **用黑盒 API 又等于没懂。** 调 `llama-load-model-from-file` 能跑，但你并不知道 GGUF 到底怎么解析、权重怎么映射进内存、矩阵怎么前向计算。

所以我们换一条路：**按 llama.cpp 的真实分层，自己从零写出一个尽量小的简化版**。每写一层，就对照着读对应的上游源码，把"为什么这么设计"讲清楚。

> 比起"会跑"，我们更在意"讲得清楚原理"。这也是整个项目取舍的准则。

---

## 方法 Methods

1. **从零实现，不抄上游**：每一章自己写最小实现，刻意保持小、只关注该章那一个原理。
2. **对照真实源码**：每个实现都标注对应 `llama.cpp/` 内的实际源码位置（相对路径），以仓库里真实代码为准，不凭记忆。
3. **轻量可跑**：用 tinybrainbot（约 200MB，F16）当测试模型，8GB 内存也能跑；测试不引入第三方框架。
4. **阶段渐进**：严格分阶段，只做当前阶段的事，不越界。

---

## 仓库结构 Layout

```
llama.cpp-101/
├── llama.cpp/                     # 上游 llama.cpp 完整 clone（被研究对象，git 忽略）
├── resources/                     # 测试用模型权重（tinybrainbot 等，git 忽略）
├── 00-what-is-llama-cpp/          # 00 章：llama.cpp 是什么
├── 01-load-and-check-gguf/        # 01 章：加载并校验 GGUF（裸 GGUF 解析器）
└── AGENTS.md                      # 协作者/贡献者须知
```

> `llama.cpp/`、`resources/`、`.omo/` 均在上传时被忽略；每个章节自带源码、测试与 Makefile。

---

## 阶段路线 Roadmap

按 llama.cpp 的真实分层，把从"读文件"到"跑推理"拆成三阶段：

| 阶段 | 内容 | 对应上游 | 状态 |
|------|------|----------|:----:|
| ① `gguf_context` | 只读 GGUF 元数据：header、KV、tensor info、对齐、bounds 校验 | `gguf.cpp::gguf_init_from_file` | ✅ 01 章完成 |
| ② `llama_model_loader` | 用 tensor 信息建 `ggml_tensor`、换算 offs、mmap/加载权重 | `llama-model-loader` | ⬜ 02 章 |
| ③ 推理 | 前向计算、采样、KV 缓存 | ggml 计算图 + `llama_context` | ⬜ 03 章 |

快速开始：`cd 01-load-and-check-gguf && make run`（默认解析 tinybrainbot 的 GGUF 并校验）。

---

<a name="english"></a>

## English

**llama.cpp-101** is a tutorial that explains llama.cpp's internals from the ground up. Instead of treating llama.cpp as a black box, we re-implement each of its layers in a minimal form, then read the actual upstream source to understand *why* it's designed that way.

**Why built this way:**

- Reading llama.cpp directly is overwhelming — production C++ with mmap, multi-shard, endianness, quantization kernels. For a beginner it's nearly unreadable.
- Using the high-level API (`llama_load_model_from_file`) runs, but teaches you nothing about how GGUF is parsed, how weights are mapped into memory, or how the matmul forward pass works.

So we write, by hand, a minimal version of each layer, and annotate where in the real `llama.cpp/` source each piece lives.

**Goals:** minimal, runnable, principle-first. Test model is tinybrainbot (~200MB, F16, runs on 8GB RAM). No third-party test framework.

**Roadmap:**

| Phase | Scope | Upstream counterpart | Status |
|-------|-------|----------------------|:------:|
| ① `gguf_context` | Read GGUF metadata: header, KV, tensor info, alignment, bounds checks | `gguf.cpp::gguf_init_from_file` | ✅ Ch.01 |
| ② `llama_model_loader` | Create `ggml_tensor`, compute offs, mmap/load weights | `llama-model-loader` | ⬜ Ch.02 |
| ③ Inference | Forward pass, sampling, KV cache | ggml graph + `llama_context` | ⬜ Ch.03 |

**Layout:** `llama.cpp/` (upstream clone, git-ignored), `resources/` (test weights, git-ignored), `00-what-is-llama-cpp/`, `01-load-and-check-gguf/` (completed Ch.01), plus `AGENTS.md` for contributors.

Get started: `cd 01-load-and-check-gguf && make run`.
