# llama.cpp-101

> 从零讲清 llama.cpp 的底层原理：按真实分层，一步步写出一个**小而精的推理引擎**。每章对照上游源码，把「为什么这么设计」讲明白。

## 为什么有这个项目

llama.cpp 源码体量过大——几十种量化、多后端（CPU/CUDA/Metal/Vulkan）、mmap/多分片/端序等大量"必要复杂度"，容易让人迷失。

本项目的做法：**按 llama.cpp 的真实分层，自己动手写最小实现**，每层对照上游源码讲清原理，最后汇聚成一个可运行的推理项目。重在「讲得清原理」，而非交付一个产品。

## 章节进度

最新分章粒度以 [`ROADMAP.md`](ROADMAP.md) 为准。当前各章目录：

| 章  | 目录                       | 主题                                      | 状态 |
| --- | -------------------------- | ----------------------------------------- | :--: |
| 00  | `00-what-is-llama-cpp/`    | llama.cpp 是什么                          |  ✅  |
| 01  | `01-load-and-check-gguf/`  | 加载并校验 GGUF（裸解析器）               |  ✅  |
| 02  | `02-ggml-context/`         | 迷你 ggml 数据结构层                      |  ✅  |
| 03  | `03-ggml-build-context/`   | 迷你 ggml 加载层（池子分配函数）          |  ✅  |
| 04  | `04-aggregate-functions/`  | 文件 IO 封装层（llama_file + llama_mmap） |  ✅  |
| 05  | `05-llama-model-load/`     | 建 llama_model 聚合对象 + 加载权重        |  ✅  |
| 06  | `06-llama-model-assembly/` | 模型语义（HParams + Layer/Model 组装）    |  ✅  |
| 07  | `07-llama-model-vocab/`    | 词表 Vocab（tokenize / detokenize）       |  ✅  |
| 08  | `08-llama-model-kernel/`   | 算子内核（kernel 层）                     |  ✅  |
| 09  | `09-llm-graph/`            | ggml 计算图设施                           |  ✅  |
| 10  | `10-llama-model-graph/`    | 整体模型图（llama 前向建图）              |  ✅  |
| 11  | `11-llama-model-run-it!/`  | 自回归采样 + 跑起来                       |  ✅  |
| 12  | —                        | 算子优化（多线程）                        | ⬜ 计划中 |

> 各章的概念与依赖顺序见 `ROADMAP.md`；当前已规划到 **12 章「算子优化（多线程）」**（⬜ 计划中）。

## 快速开始

01–07 用 Makefile（`make run`），08–11 用 CMake。测试模型用 `resources/` 下的 tinybrainbot（约 200MB，F16，git 忽略）。

### ① 下载模型（HuggingFace）

测试模型：**https://huggingface.co/nkthebass/tinybrainbot-100m-v3-instruct**

```bash
# 用 huggingface-cli 下载 GGUF 权重到 resources/ 下
huggingface-cli download nkthebass/tinybrainbot-100m-v3-instruct \
  tinybrainbot-100m-v3-instruct-f16.gguf \
  --local-dir resources/tinybrainbot-100m-v3-instruct
```

### ② 跑起来（需要模型）

```bash
# 01：裸 GGUF 解析器，解析并校验 tinybrainbot
cd 01-load-and-check-gguf && make run

# 05：完整加载 llama_model（110 个 tensor，mmap 零拷贝）
cd 05-llama-model-load && make run

# 07：词表 Vocab，中文/英文 tokenize<->detokenize 往返
cd 07-llama-model-vocab && make run

# 11：完整前向 → 自回归采样，生成一整段文本
cd 11-llama-model-run-it! && cmake -B build -S . && cmake --build build && ./build/llama-cli <path/to/model.gguf> -n 64
```

### ③ Kernel / 图（不依赖模型）

```bash
# 08：算子内核测试（不依赖模型）
cd 08-llama-model-kernel && cmake -B build-kernel -S src/kernel && cmake --build build-kernel && ./build-kernel/test-kernel

# 09：ggml 计算图设施测试（不依赖模型）
cd 09-llm-graph && cmake -B build -S . && cmake --build build && ./build/test-ggml-graph
```

## 仓库结构

```
llama.cpp-101/
├── llama.cpp/                     # 上游 llama.cpp 完整 clone（被研究对象，git 忽略）
├── resources/                     # 测试用模型权重（tinybrainbot 等，git 忽略）
├── 00-what-is-llama-cpp/          # 00 章
├── 01-load-and-check-gguf/        # 01 章
├── ...                            # 02–11 章（见上方进度表）
├── README.md                      # 本文件
├── LICENSE                        # MIT 协议
├── ROADMAP.md                     # 权威分章路线图（改分章前先读它）
├── AGENTS.md                      # 协作者 / 贡献者须知
└── playground/                    # 随手实验代码（git 忽略）
```

> `llama.cpp/`、`resources/`、`.omo/`、`playground/` 均被 git 忽略；每个可构建章节自带源码、测试与构建配置（04+ 自包含，含前几章代码的本地拷贝）。

## 方法

1. **从零实现，不抄上游**：每章写最小实现，只讲清该章那一个原理。
2. **对照真实源码**：每个实现标注对应 `llama.cpp/` 的实际源码位置（相对路径），以真实代码为准。
3. **轻量可跑**：用 tinybrainbot（约 200MB，F16）当测试模型；测试不引入第三方框架。
4. **最终有产出**：汇聚成一个带服务 API 的推理项目。

## License

本项目以 **MIT 协议**开源，详见 [`LICENSE`](LICENSE)。

> 本项目为教学用途，代码为从零实现、对照上游理解；参考的 `llama.cpp` 亦以 **MIT** 协议开源（见其仓库 LICENSE）。各章节标注的上游源码位置仅供学习对照，不在本项目内复制其代码。
