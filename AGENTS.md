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

> 章节被拆分重排过：先 `01`（GGUF 解析），再迷你 ggml 拆成 **02 结构 / 03 加载 / 04 llama_model** 三章。旧 AGENTS 里「02 直接建 llama_model」的说法已过时。

- `00-what-is-llama-cpp/` —— 00 章「什么是 llama.cpp」（`README.md`，已完成）。
- `01-load-and-check-gguf/` —— 01 章「加载并校验 GGUF」：裸 GGUF 解析器（只读元数据；`gguf_context`/`gguf_tensor_info`/`gguf_kv` 对齐上游）。已完成、有 `Makefile` 与手写测试。
- `02-ggml-context/` —— **02 章「迷你 ggml 数据结构层」**：`namespace ggml` 里的 `ggml_type`/`ggml_tensor`/`ggml_context`(前向声明)/`ggml_object_type`/`ggml_cgraph`(仅为 03 预留声明) 的**类型与布局**。
  - ⚠️ 本章**不写任何函数**（无 `ggml_init`/`ggml_new_tensor_*`），`src/ggml.cpp` 只有结构定义；`test/` 目录当前为空、未纳入 git。
  - 文件：`include/ggml.h`（类型层）、`src/ggml.cpp`（`ggml_object`/`ggml_context` 内部定义）。
- `03-ggml-load/` —— **03 章「迷你 ggml 函数/加载层」**（把池子分配 API 写出来：`ggml_init`/`ggml_new_object`/`ggml_new_tensor_*`/`ggml_set_name`/`ggml_nbytes`）。🚧 **尚未实现**，目前只有 `reference.md`。
- `04-llama-model/` —— **04 章「建 `llama_model`」**（用 03 的能力把 GGUF 上百个 tensor 逐个实例化 + mmap 零拷贝挂数据）。🚧 **尚未实现**，目前只有 `reference.md`。原 `03-llama-model/` 已重命名为此目录。
- `resources/` —— 测试用模型权重（git 忽略）。默认模型：**`tinybrainbot-100m-v3-instruct/tinybrainbot-100m-v3-instruct-f16.gguf`**（约 200MB，F16/Llama，本教程所有推理都用它，8GB 内存跑得动）。另留 `Bonsai-27B-Q1_0.gguf`（3.8GB，Q1_0 量化对照）。都**别复制/移动/提交**，引用用相对路径。
- `ROADMAP.md` —— 面向 30–31 章的**权威分章路线图**（draft）。改章节粒度/加章前先读它，别凭旧 AGENTS 的 ②③ 设想。
- `playground/` —— 随手实验代码（`a.out`/`g.c` 等），**git 忽略**，不是成稿。
- `.omo/` —— 编排/延续运行机制，**不要编辑**。

## 关键技术事实（实测，勿凭记忆改）

- 机器：macOS（darwin），**64 位小端**，GGUF 固定小端，直接 `memcpy` 读即可。**本机内存仅 8GB**，跑 27B 或大编译会吃紧。
- tinybrainbot 实测：magic=`GGUF`、version=3、n_tensors=110、n_kv=32、architecture=`llama`、type=`model`、alignment=默认 32、data_offset=0xbaf80、file_size=200989568。tensor 类型只用 `0(F32)×25` 和 `1(F16)×85`。
- Bonsai 实测：magic=`GGUF`、version=3、n_tensors=851、n_kv=37、architecture=`qwen35`、type=`model`、data_offset=0xa7bc40、file_size=3803452480。tensor 只用 `41(Q1_0)×498` 和 `0(F32)×353`。
- 量化参数（`type_traits[]`，见 `ggml/src/ggml.c` 631–760 与 `ggml-common.h` block 结构体）：Q1_0 blk=128/size=18、Q4_0 blk=32/size=18、Q4_K blk=256/size=144、Q6_K blk=256/size=210、F32 blk=1/size=4。
- 手算 nbytes 公式见 01 README 九章（`blk_size`/`type_size` 从量化 block 结构体 `sizeof` 推出）。
- 行主序 `nb[]`：`nb[0]`=每元素字节数，`nb[i]=nb[i-1]×ne[i-1]`；`nbytes=(ne[0]/blk_size)×type_size×ne[1]×ne[2]×ne[3]`。

## 陷阱 & 注意事项

- **不要混淆两个 MAGIC**：`gguf.h` 的 `GGUF_MAGIC "GGUF"`（GGUF **权重文件**）与 `ggml.h` 的 `GGML_FILE_MAGIC 0x67676d6c`/`"ggml"`（ggml **计算图**序列化）是两种不同格式的标识，非一份定义的两处拷贝。
- **clangd（LSP）诊断对 `01/02/03/04` 报"找不到头文件 / 未声明标识符"是环境假阳性**：clangd 未配置 `-Iinclude` 与交叉 `-I../01-load-and-check-gguf/include`，导致 `gguf.h`/`ggml.h` 无法解析、级联一堆 incomplete type 错误（`unused-includes` 警告亦然）。实际 `make` 用 `-std=c++11 -Wall -Wextra` 编译干净。**别被误导去"修"它。**

## 约定

- 中文表达；教程讲解性质，优先准确性与可读性，引用真实 llama.cpp 源码（`llama.cpp/` 内实际代码为准，别凭记忆写 API）。引用用相对路径（如 `llama.cpp/src/llama.cpp`），不为讲解需要不复制大段代码。
- **教程不必逐字照抄 llama.cpp**：核心是让读者对某机制（命名空间、不透明句柄、内存池）有具象认知——用真实的迷你实现把一个 C++ 概念"看得到、摸得着"。包法/命名/范围可从教学出发取舍（例：`namespace ggml` 用来直观展示命名空间拆解名字冲突）。
- 测试不引入第三方框架：手写 `test_*.cpp` + `main`，退出码非 0=失败，临时文件放 `/tmp`。
- `reference.md`（各章「参考源码对照」）**越精简越好**：只说明「大概看了 `llama.cpp/` 哪个文件、哪些函数」，不展开解析/背景，别把讲解内容塞进去。
