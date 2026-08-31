# llama.cpp-101

从零讲解 llama.cpp 的教程项目（中文）。目标是把 GGUF / 模型加载 / 推理等底层原理讲清楚。写代码 / 教程时以「如何讲清原理」为准，不是交付一个可运行的产品。

**章节进度 / 分章粒度以 `ROADMAP.md` 为准**（30–31 章草案，含每条主线对应的上游源码位置）。本文件只写「不读源码就猜不到」的仓库级事实。

## 双重 git 仓库（最易踩坑）

- **根仓库**（本目录）：remote `origin` = `git@github.com:HeisenbergUwU/llama.cpp-101.git`，分支 `master`。
- **`llama.cpp/` 是嵌套的独立 git 克隆**，与根仓库互不隶属（被根仓库 `.gitignore` 忽略）。它是**被研究对象**，不是本项目的代码。
  - 不要在 `llama.cpp/` 里做"面向本项目"的修改/提交；它自带自己的 `AGENTS.md` / `CLAUDE.md`，改代码先读那两份。
  - remote：`origin` = 上游 ggerganov/llama.cpp，`mine` = 用户 fork（HeisenbergUwU/llama.cpp），当前分支 `fix-grammar-readme`。
- **提交/推送前先 `git status` 确认在哪一个仓库里。**

## 目录结构（现状，与旧版 AGENTS 不符，勿凭旧记忆）

> 章节多次拆分重排，已不是「01→02→03→04 llama_model」的简单递增。**以本列表为准。**

- `00-what-is-llama-cpp/` —— 00 章「什么是 llama.cpp」（`README.md`，已完成，无代码）。
- `01-load-and-check-gguf/` —— 01 章「加载并校验 GGUF」：裸 GGUF 解析器（只读元数据；`gguf_context`/`gguf_tensor_info`/`gguf_kv` 对齐上游）。**仍用 `FILE*`/`fread` 顺序读，没走 llama-io**（它是解析原理的独立讲解）。
- `02-ggml-context/` —— 02 章「迷你 ggml 数据结构层」：`namespace ggml` 里的类型与布局。⚠️ **不写任何函数**（无 `ggml_init`/`ggml_new_tensor_*`），无 Makefile/测试，不可独立构建。
- `03-ggml-build-context/` —— 03 章「迷你 ggml 加载层」：把池子分配函数写出来（`ggml_init`/`ggml_new_tensor_*`/`ggml_set_name`/`ggml_nbytes`）。**已实现**，有 `Makefile` + 手写测试 `test-ggml-build-context`。目录名是 `build-context`，不是旧 AGENTS 写的 `ggml-load`。
- `04-aggregate-functions/` —— 04 章「文件 IO 封装层」：`llama-io`（`llama_file` + `llama_mmap`）。**已实现**。
- `05-llama-model-load/` —— 05 章「建 `llama_model` 聚合对象 + 加载权重」。**已实现**。
- `resources/` —— 测试用模型权重（git 忽略）。默认模型：**`tinybrainbot-100m-v3-instruct/tinybrainbot-100m-v3-instruct-f16.gguf`**（约 200MB，F16/Llama，本教程所有推理都用它，8GB 内存跑得动）。另留 `Bonsai-27B-Q1_0.gguf`（3.8GB，Q1_0 量化对照）。都**别复制/移动/提交**，引用用相对路径。
- `playground/` —— 随手实验代码（`a.out`/`g.c`/`mmap_demo.cpp` 等），**git 忽略**，不是成稿。
- `.omo/` —— 编排/延续运行机制，**不要编辑**。

## 04/05 章自包含 + 统一的文件 IO 封装（新，最容易猜错）

- **04/05 是「自包含」章**：每章 `src/` + `include/` 里都**各有一份拷贝** `gguf.h/cpp`、`ggml.h/cpp`、`llama-io.h/cpp`。`#include "gguf.h"` / `"ggml.h"` 指的是**本章自己的拷贝**，不是 01/03 的文件。Makefile 只用 `-Iinclude`（本章目录内），**不跨章 `-I`**。
- **`llama-io` 是 04/05 唯一的文件 IO 入口**：`llama_file`（open/fstat/顺序读/fclose）+ `llama_mmap`（mmap/munmap）。04/05 里的 `gguf.cpp`、`llama-model.cpp` **不再裸调任何系统调用**——所有 `open/mmap/munmap/close/fopen` 只存在于 `llama-io.cpp` 内部。
- **01 的 gguf.cpp 是例外**：仍是原始 `FILE*`/`fread` 版，没走 llama-io（01 是独立解析原理章）。
- **`llama_mmap` 禁拷贝、可移动**：它持独占的 mmap 地址，`llama_model`(05) 持有它时靠移动语义（`llm.mmap = std::move(mapping)`，把局部 `mapping` 移入 `llm.mmap`）。移动构造/移动赋值已实现于 `llama-io.cpp`，**约定：接管后必须把源 `addr` 置空，否则两对象析构会对同一地址双 `munmap`**。改这段时别破坏这个不变量。
- **05 的 ggml 池子大小是按 tensor 个数动态算的**（`pool_size = 对象头 + n_tensors×(对象头+align16(sizeof ggml_tensor)) + 余量`）。`ggml_init` 在 `gguf_load` **之后**（要先拿到 `n_tensors`）。⚠️ **别用 `mapping.size`（= 文件大小 ~200MB）当池子**——`no_alloc=true` 池子只装结构，110 个 tensor 只需约 22KB，198MB calloc 是浪费。
- **每章编译产物二进制被 `.gitignore` 忽略**（`04-aggregate-functions/test-*`、`05-llama-model-load/test-*`、旧 `04-llama-model-load/test-*`）。提交时别 `git add` 二进制。

## 关键技术事实（实测，勿凭记忆改）

- 机器：macOS（darwin），**64 位小端**，GGUF 固定小端，直接 `memcpy` 读即可。**本机内存仅 8GB**，跑 27B 或大编译会吃紧。
- tinybrainbot 实测：magic=`GGUF`、version=3、n_tensors=110、n_kv=32、architecture=`llama`、type=`model`、alignment=默认 32、data_offset=0xbaf80、file_size=200989568。tensor 类型只用 `0(F32)×25` 和 `1(F16)×85`。
- Bonsai 实测：magic=`GGUF`、version=3、n_tensors=851、n_kv=37、architecture=`qwen35`、type=`model`、data_offset=0xa7bc40、file_size=3803452480。tensor 只用 `41(Q1_0)×498` 和 `0(F32)×353`。
- 量化参数（`type_traits[]`，见 `ggml/src/ggml.c` 631–760 与 `ggml-common.h` block 结构体）：Q1_0 blk=128/size=18、Q4_0 blk=32/size=18、Q4_K blk=256/size=144、Q6_K blk=256/size=210、F32 blk=1/size=4。
- 手算 nbytes 公式见 01 README 九章（`blk_size`/`type_size` 从量化 block 结构体 `sizeof` 推出）。
- 行主序 `nb[]`：`nb[0]`=每元素字节数，`nb[i]=nb[i-1]×ne[i-1]`；`nbytes=(ne[0]/blk_size)×type_size×ne[1]×ne[2]×ne[3]`。

## 陷阱 & 注意事项

- **不要混淆两个 MAGIC**：`gguf.h` 的 `GGUF_MAGIC "GGUF"`（GGUF **权重文件**）与 `ggml.h` 的 `GGML_FILE_MAGIC 0x67676d6c`/`"ggml"`（ggml **计算图**序列化）是两种不同格式的标识，非一份定义的两处拷贝。
- **声明"类型同名"的对象时用 `{}` 不要 `()`**：如 `gguf::gguf_context gguf_context();` 的空括号会被 C++ 当成**函数声明**（最恼人的解析 most vexing parse）→ 传给 `gguf_load(..., gguf_context, ...)` 报"cannot bind"，编译器还有 `-Wvexing-parse` 警告。写 `gguf::gguf_context gguf_context{};`（或 `= {}`）。项目里 `05` 常见这种"变量名 = 类型名"的写法（`gguf_context` / `llama_file`），合法但必须用 `{}` 构造。
- **clangd（LSP）诊断对 `01/02/03/04/05` 报"找不到头文件 / 未声明标识符"是环境假阳性**：clangd 未配置各章的 `-Iinclude`，导致 `gguf.h`/`ggml.h`/`llama-io.h` 无法解析、级联一堆 incomplete type 错误（`unused-includes` 警告亦然）。实际 `make` 用 `-std=c++11 -Wall -Wextra` 编译干净。**别被误导去"修"它。**

## 约定

- 中文表达；教程讲解性质，优先准确性与可读性，引用真实 llama.cpp 源码（`llama.cpp/` 内实际代码为准，别凭记忆写 API）。引用用相对路径（如 `llama.cpp/src/llama.cpp`），不为讲解需要不复制大段代码。
- **教程不必逐字照抄 llama.cpp**：核心是让读者对某机制（命名空间、不透明句柄、内存池）有具象认知——用真实的迷你实现把一个 C++ 概念"看得到、摸得着"。包法/命名/范围可从教学出发取舍（例：`namespace ggml` 用来直观展示命名空间拆解名字冲突）。
- 测试不引入第三方框架：手写 `test_*.cpp` + `main`，退出码非 0=失败，临时文件放 `/tmp`。
- `reference.md`（各章「参考源码对照」）**越精简越好**：只说明「大概看了 `llama.cpp/` 哪个文件、哪些函数」，不展开解析/背景，别把讲解内容塞进去。
