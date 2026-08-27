# llama.cpp-101

这是一个 **从零讲解 llama.cpp 的教程项目**（中文）。目标是帮助读者理解 GGUF、模型加载、推理等底层原理。写代码 / 教程内容时，以“如何讲清楚原理”为准，而不是提交一个完整可运行的产品。

## 目录结构（易踩坑）

> **本仓库根目录自身是一个 git 仓库**（remote `origin` = `github.com:HeisenbergUwU/llama.cpp-101.git`，分支 `master`）。`llama.cpp/` 是**嵌套的独立 git 克隆**，与根仓库互不隶属。提交 / 推送前先 `git status` 确认当前在哪个仓库里。

- `llama.cpp/` —— **上游 llama.cpp 源码的完整 git 克隆**（被研究对象，不是本项目的代码，被根仓库 ignore）。
  - 不要在这个子目录里做“面向项目”的修改或提交；它自带自己的 `AGENTS.md` / `CLAUDE.md`，改代码请先读那两份。
  - remote：`origin` = 上游 ggerganov/llama.cpp，`mine` = 用户自己的 fork（HeisenbergUwU/llama.cpp）。当前在分支 `fix-grammar-readme`。
  - 引用源码时，用相对路径写（例如 `llama.cpp/src/llama.cpp`），不要复制粘贴大段代码进教程，除非为讲解需要。
- `resources/` —— 存放测试用材料（模型权重等）。
  - 默认测试/讲解模型：**`tinybrainbot-100m-v3-instruct/tinybrainbot-100m-v3-instruct-f16.gguf`**（约 200MB，F16/Llama 架构，本教程 01 章与后续推理都用它，体积小、8GB 内存跑得动）。
  - 另外还留有一个 **3.8GB 的 `Bonsai-27B-Q1_0.gguf`**（Q1_0 量化对照用）。这两个都是大/重文件，不要复制、移动或提交到别处；教程里引用它们时用相对路径。
- `00-what-is-llama-cpp/` —— 00 章「什么是 llama.cpp」，已写完（`README.md`）。
- `01-load-and-check-gguf/` —— **01 章「加载并校验 GGUF（load + check）」**，已实现并验证通过。
  - `include/gguf.h`、`src/gguf.cpp`：章节实现，**裸 GGUF 解析器**（只读元数据，不引入 ggml tensor/context 对象；类型名对齐上游：`gguf_context`/`gguf_tensor_info`/`gguf_kv`）。
  - `tests/test-load-gguf.cpp`：手写测试入口（无第三方框架，退出码非 0=失败）。
  - `Makefile`：`make` / `make run` / `make clean`，默认 MODEL=`../resources/tinybrainbot-100m-v3-instruct/tinybrainbot-100m-v3-instruct-f16.gguf`。
  - `README.md`：含解析流程、校验项表、tinybrainbot 类型实测、参考资料（上游源码清单）。
- `02-model-loader/` —— **02 章「建 `llama_model`（llama_model + 迷你 ggml + 真实 ggml_tensor）」**，已实现并验证通过。
  - `include/ggml.h` + `src/ggml.cpp`：**迷你 ggml**（`ggml_context`/`ggml_tensor`/`ggml_object`/池子式内存分配/`ggml_new_tensor_*`/`nb[]` 换算），字段与算法对齐上游 `ggml/src/ggml.c`（新版 `objects_begin/end` + `mem_buffer_owned`）；放 `ggml` 命名空间（避免与 01 章全局 `GGML_TYPE_*` 冲突），类型名本身上游一致。
  - `include/llama-model.h` + `src/llama-model.cpp`：**核心 `llama_model` 纯 struct 聚合对象**（对齐上游 `llama-model.h`），持 `gguf::gguf_context` + `ggml_context *`（no_alloc=true）+ `std::vector<ggml_tensor*>` + `tensors_by_name` + mmap 基址；`llama_model::load` = 解析 GGUF -> 建 ctx -> 逐个建 110 个真实 `ggml_tensor` -> mmap 零拷贝挂 `data`。
  - `tests/test-llama-model.cpp`：手写测试（无第三方框架，退出码非 0=失败），校验 110 个 tensor 的 ne/nb/type/name + mmap 零拷贝数据一致性。
  - `Makefile`：`make` / `make run` / `make clean`，默认 MODEL=`../resources/tinybrainbot-100m-v3-instruct/tinybrainbot-100m-v3-instruct-f16.gguf`；复用 01 章的 `../01-load-and-check-gguf/src/gguf.cpp`。
  - `README.md`：教程说明（写实现时再更新到最终形态）。
- `03-*/`、`04-*/` —— 尚未创建，逐章再定，不预设。
- `.omo/` —— 编排/延续运行机制，**不要编辑**。

## 阶段边界（实现路线）

> 已完成 01、02 章。**03 起逐章再定，不预设**；一次做一步，不大不小，做到"正好能理解"的程度就停，不向下游越界。

1. **① `gguf_context`**（01 章，已完成）：只读 GGUF 元数据——header、KV、tensor info、对齐、bounds 校验，类型名对齐上游（`gguf_context`/`gguf_tensor_info`/`gguf_kv`）。不加载上下文、不创建 tensor。
2. **② `llama_model`**（02 章，已完成）：**建一个聚合对象 `llama_model`**，把所有东西装进去——内部持有迷你 `ggml_context`，用**池子式内存分配**把 110 个权重建成**真实的 `ggml_tensor` 对象**（`ne[]`/`nb[]`/`type`/`name`，no_alloc=true，mmap 零拷贝拖数据）。**范围只到"持有 tensor 对象"，不建图、不执行**（`ggml_mul_mat`/`ggml_cgraph`/CPU 执行是后续阶段，避免一次太满）。
3. ③ 推理等 —— 尚未开始，逐章再定。

## 关键技术事实（来自实测，勿凭记忆改）

- 机器：macOS（darwin），**64 位小端**，GGUF 固定小端，直接 `memcpy` 读即可。**本机内存仅 8GB**，跑 27B 或大编译会吃紧。
- tinybrainbot 文件实测：magic=`GGUF`、version=3、n_tensors=110、n_kv=32、architecture=`llama`、type=`model`、alignment=默认 32、data_offset=0xbaf80 (765824)、file_size=200989568。tensor 类型只用两种：`0(F32)×25` 和 `1(F16)×85`。
- Bonsai 文件实测（量化对照）：magic=`GGUF`、version=3、n_tensors=851、n_kv=37、architecture=`qwen35`、type=`model`、alignment=默认 32、data_offset=0xa7bc40 (10992704)、file_size=3803452480。tensor 类型只用两种：`41(Q1_0)×498` 和 `0(F32)×353`。
- 量化类型参数（取自 llama.cpp 内部 `type_traits[]` 表，可在 `ggml/src/ggml.c` 631-760 及 `ggml/src/ggml-common.h` 的 block 结构体核对）：Q1_0 blk=128/size=18、Q4_0 blk=32/size=18、Q4_K blk=256/size=144、Q6_K blk=256/size=210、F32 blk=1/size=4。
- 手算 nbytes 公式见 01 README 九章（`blk_size`/`type_size` 是从量化 block 结构体 `sizeof` 推出来的）。

## 陷阱 & 注意事项

- **不要混淆两个 MAGIC**：`gguf.h` 的 `GGUF_MAGIC "GGUF"`（标识 GGUF **权重文件**）与 `ggml.h` 的 `GGML_FILE_MAGIC 0x67676d6c`/`"ggml"`（标识 **ggml 计算图**序列化）是两种不同格式的标识，非一份定义的两处拷贝。
- 本机 LSP（clangd）诊断对 `01-load-and-check-gguf/` 和 `02-model-loader/` 报"找不到头文件 / 未声明标识符"是**环境假阳性**（clangd 未配置 `-Iinclude` 和 `-I../01-load-and-check-gguf/include`，导致 `gguf.h`/`ggml.h` 无法解析、级联出一堆 incomplete type 错误；`unused-includes` 警告亦然）。实际 `make` 用 `-std=c++11 -Wall -Wextra` 编译干净。别被误导去"修"它。

## 约定

- 操作系统是 macOS（darwin），延续中文表达风格。现有指令文件是中文；后续教程内容也写中文。
- 教程是讲解性质：优先保证准确性和可读性，引用真实的 llama.cpp 源码（以 `llama.cpp/` 内的实际代码为准，不要凭记忆写 API）；引用时用相对路径（如 `llama.cpp/src/llama.cpp`），不复制大段代码进教程，除非为讲解需要。
- **教程不一定逐字照抄 llama.cpp**：核心是让读者对某个机制（如命名空间、不透明结构/句柄、内存池）有**具象化认知**——用真实的迷你实现把一个 C++ 概念"看得到、摸得着"。因此讲某机制时，只要实现了那个原理，包法/命名/范围可以从教学出发取舍，不必与上游完全一致（例如 `namespace ggml` 用来让读者直观看到命名空间拆解名字冲突，而不是纯粹照抄上游的布局）。
- 测试不引入第三方框架：手写 `test_*.cpp` + `main`，退出码非 0=失败，临时文件放 `/tmp`。
- **`reference.md`（各章的「参考源码对照」）写得越精简越好**：只需说明「大概看了 `llama.cpp/` 里的哪个文件、哪些函数」即可，不用展开详细解析/叙述背景，不要把讲解内容塞进去。
