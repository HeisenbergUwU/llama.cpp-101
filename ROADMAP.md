# llama.cpp-101 路线图（草案 · 31 章）

> **状态：草案，待审阅**。目标是把整条链路拆到「CPU + Metal 的 server 推理」。
> 每一章对应一个**可独立讲、可手写代码 + 手写测试**的概念，且**有真实上游源码可对照**。
> 用 tinybrainbot（plain llama，F16/F32）当测试模型——它不涉及 MoE/SWA/flash-attn，正好能走通一条最小的「必须讲」主线。
>
> 上游文件路径以 `llama.cpp-101/llama.cpp/` 为根（相对路径）。文件名/函数已按当前 clone 核对；为稳健起见，文档只精确到「概念 ↔ 文件 ↔ 关键函数」，不绑易漂移的行号。

---

## 🎯 精简主线（当前优先级：先做到「CPU server 能 chat completion 一个小模型」）

> 下方的大路线图是长期参考；**这里才是最先要走的快路径**。基于 tinybrainbot 实测
> （GQA、无 bias、无独立 lm_head 复用 token_embd、SwiGLU-PAR、RMS eps=1e-5、
> RoPE linear base=10000/rot=64），把「CPU server 出文本」压缩到最少章节。
>
> **server-first 策略（2026-09 拍板）**：先搭一条「前向→采样→server」的最短链路把里程碑拿下，
> **KV cache / batch 切分延后**为优化章节（11/12）。MVP 阶段前向**全量重算、不引入 KV cache**
> （每次喂全部历史 token 重算 attention），先把链路跑通。

| 章 | 内容 | 产出 |
|----|------|------|
| **02–05** | 基础加载（✅ 全部完成）：**02** 迷你 ggml 数据结构；**03** 迷你 ggml 函数/加载层（`ggml_init` 建池 / `ggml_new_tensor_*` 实例化 / `nb[]` 换算 / `ggml_nbytes`）；**04** 文件 IO 封装（`llama_file` + `llama_mmap`）；**05** `llama_model`（组装 110 个裸 tensor + mmap 零拷贝拖权重） | 拿到「能按名取、数据零拷贝」的模型 |
| **06** | 模型语义：HParams + Layer/Model 组装（✅） | 「有语义」的模型 |
| **07** | 词表 Vocab：tokenize / detokenize（✅） | 文字 ↔ token id 双向 |
| **08** | **完整前向（全量重算、无 KV）**（✅）：RMSNorm→QKV→RoPE→注意力→FFN→输出 norm→lm_head→logits；argmax 采 1 token | 给一个 prompt 算出 logits + 贪心 1 token |
| **09** | 自回归采样：贪心循环生成多 token，直到 EOG / max_tokens | 命令行能生成一段文本 |
| **10** | 最小 CPU server：HTTP + `POST /v1/chat/completions`（非流式先出 JSON） | **curl 一发，拿到回复文本** |
| 11（优化） | KV cache（从 08 的 `KVSlice` 接缝换入真缓存） | 长上下文不重算 |
| 12（优化） | batch / ubatch 切分（多序列） | 批处理 |

> 先打卡「chat completion 出文本」这个里程碑，其余（KV cache / batch / quant / Metal）
> 作为后续优化章节再接。

---

## 阶段总览（完整路线，长期参考）

| 阶段 | 章节 | 主题 | 里程碑 |
|------|------|------|--------|
| ①② 基础 | 00–05 | 什么是 / GGUF 解析 / ggml 数据结构 / ggml 加载 / 文件IO / llama_model | ✅ 00–05 完成 |
| ③ 模型语义层 | 05–07 | hparams → llama_model 语义 → vocab | ✅ 拿到「语义模型」 |
| ③.5 server 快路径 | 08–10 | 完整前向 → 采样 → 最小 server | 🎯 **CPU server 出文本** |
| ④ 上下文层(优化) | 11–12 | KV cache / batch 切分 | 运行期内存与批处理 |
| ⑤ 前向算子深挖 | 13–22 | transformer 逐算子拆解（在 08 基础上深化） | 每算子可独立讲/测试 |
| ⑥ 计算执行 + backend | 23–26 | 图执行 / backend 抽象 / CPU 内核 | **CPU 能跑出 logits → token** |
| ⑦ Metal backend | 27–29 | 接同一条分发表 / 内核 / server 切 Metal | **CPU+Metal 双后端 server** |
| ⑧ 收尾 | 30 | 汇总结论 | 全书收官 |

> 编排说明：**先走 08–10 的 server 快路径把 CPU server 跑通**（里程碑），再把 KV cache / batch
> 作为优化章节（11–12）补上；随后把前向逐算子深挖（13–22）置于已跑通的前向之上（每章深化一个
> ch08 刻意精简的算子），最后用 backend 抽象（23–26）+ Metal（27–29）插成第二后端——正好演示
> 「为什么当初要留 backend 槽位」。

---

## ①② 基础（已完成 / 进行中）

- **00** `00-what-is-llama-cpp/` — 什么是 llama.cpp。✅
- **01** `01-load-and-check-gguf/` — 裸 GGUF 解析器（header / KV / tensor info / 对齐 / bounds）。✅
- **02** `02-ggml-context/` — 迷你 ggml 的**数据结构层**：`ggml_type` / `ggml_tensor` / `ggml_context` / `ggml_object` / 池子布局 / cgraph 声明。✅（只定型与池子内部布局，**还没写任何加载函数**）
- **03** `03-ggml-build-context/` — 迷你 ggml 的**函数 / 加载层**：补上 02 没实现的池子分配 API——`ggml_init`（建池）、`ggml_new_object`（池子切块）、`ggml_new_tensor_*`（实例化张量）、`nb[]` 换算、`ggml_set_name`、`ggml_nbytes`。✅ 已实现（有 `Makefile` + 手写测试）。
- **04** `04-aggregate-functions/` — 文件 IO 封装层：`llama_file`（open/fstat/顺序读/fclose）+ `llama_mmap`（mmap/munmap）。✅ 已实现（04/05 的所有文件操作统一走这一层）。
- **05** `05-llama-model-load/` — 建聚合对象 `llama_model`：用 03 的能力把 GGUF 的 110 个 tensor 逐个实例化成真实 `ggml_tensor`（no_alloc=true）+ mmap 零拷贝挂 `data`（由 `llama_mmap` 提供）。✅ 已实现。
  - 关键：把权重「拿到内存」**不需要任何计算算子**（纯 memcpy/指针 + 池子指针），算子是 08（快路径）/ 13+（深挖）的后话。

---

## ③ 模型语义层（05–07）

从 04 的「加载出来的裸张量」升级成「有语义的模型」。

- **05 hparams：哪些 KV 决定架构**
  - 读 `llama.cpp/src/llama-hparams.h` `struct llama_hparams` + `llama-hparams.cpp` 的 `n_head()/n_layer()/n_embd_head_k…` 派生方法。
  - 从 GGUF KV 读出 n_embd、n_layer、n_head(_kv)、n_ff、rope 参数、norm eps…并算派生维度。
  - 产出：`HParams` + 解析 + 核对 tinybrainbot 实测值 + 测试。
- **06 llama_model：tensor 组装成语义层**
  - 读 `llama.cpp/src/llama-model.h`（`llama_model` + `llama_layer`）+ `llama-model.cpp` 的 `create_tensor`。
  - 把 04 的张量挂到「每层一个 bag（attn_norm/wq/wk/wv/wo/ffn_norm/gate/up/down）」+ 根张量（tok_embd/output_norm/output）。
  - 产出：`Model` 语义对象 + 层数/每层张量完备性测试。
- **07 vocab + 分词/去分词**
  - 读 `llama.cpp/src/llama-vocab.h/.cpp`（`llama_vocab`，`llama_tokenize`/`llama_token_to_piece`/`llama_vocab_is_eog`）。
  - 从 `tokenizer.ggml.*` KV 原始字节建词表，实现 tokenize（文字→id）与 detokenize（id→文字）。
  - 产出：`Vocab` + round-trip 测试（token↔text）。✅

---

## ③.5 server 快路径（08–10）——先跑通「一条」完整链路

> 这一段的宗旨：**先用最短链路让 CPU server 出文本**，把里程碑拿下；KV cache / batch / 逐算子深挖
> 都放到 11 之后。前向结构经 Oracle 裁定采用**方案 A（直写循环、不建图）**，但**从第一天预留三个接缝**，
> 让后续 KV cache 章节能"插进去不重写"。

- **08 完整前向（全量重算、无 KV cache）**
  - 读 `llama.cpp/src/models/llama.cpp`（`graph<false>::graph()`，标准 llama 前向）+ `src/llama-graph.cpp` 的 `build_norm`/`build_attn_mha`/`build_ffn`/`build_inp_embd`。
  - 直写循环：RMSNorm → QKV matmul → 头 reshape + RoPE(Q,K) → 朴素注意力（逐位置 Q·Kᵀ、scale 1/√64、因果 softmax、·V、头重组）→ wo matmul → 残差 add → SwiGLU-PAR → 输出 norm → lm_head（tied 到 token_embd）→ logits。
  - tinybrainbot 实测 hparams：n_embd=768、n_layer=12、n_head=12、n_head_kv=4（GQA=3）、n_embd_head=64、n_rot=64、n_ff=2048、rope_base=10000、eps=1e-5、SwiGLU-PAR、无 bias、tied lm_head。
  - **三个接缝（ch11 换 KV 时只动这一处）**：
    - `KVSlice`：不透明的「所有历史 KV 行」视图（k/v 形状 `[n_embd_head, n_head_kv, n_kv]` + `n_kv`）。ch08 用重算填充，ch11 换真缓存；注意力只读 `KVSlice.n_kv`，不假设它等于当前 token 数。
    - `build_causal_mask`：独立函数。ch08 产 `[n_tokens×n_tokens]`，ch11 产 `[n_kv×n_tokens]`，签名不变。
    - `forward(model, tokens, KVSlice&, positions)`：单 decode 步入口；「append K/V」藏在 forward 内部，使缓存替换对调用方不可见。
  - op 函数以裸 `float*` 提供（`rms_norm`/`matmul`/`swiglu`/`rope_inplace`/`softmax_row`），逐算子深挖章（13+）就地加深、不重写调用图。
  - 产出：`Forward`（出 logits）+ 与手算/形状对照测试。
- **09 自回归采样（greedy）**
  - 读 `llama.cpp/src/llama-sampling.h/.cpp`（`llama_sampler`，greedy 分支）+ `llama_vocab::is_eog`。
  - 贪心循环生成多 token，直到 EOG（`<eos>`）或 max_tokens。**终止逻辑做成可复用组件**（ch10 server 每请求都调它）。
  - 产出：命令行能生成一段文本 + 终止行为测试。
- **10 最小 CPU server**
  - 读 `llama.cpp/tools/server`（`llama-server`）+ `common/` 的 HTTP 层。
  - HTTP + `POST /v1/chat/completions`，非流式先出 JSON。chat template 用 10 行硬编码（如 `[INST]…[/INST]`），**不让 jinja 复杂度爬进里程碑**。
  - ⚠️ HTTP 库（cpp-httplib vs 手写 socket）**延后到本章再定**。
  - 产出：**curl 一发，拿到回复文本**（里程碑）。

---

## ④ 上下文层（优化，11–12）——运行态内存与「一次喂多少 token」

> KV cache / batch 延后到 server 快路径之后。ch08 的三个接缝（`KVSlice`/`build_causal_mask`/`forward`）
> 正是为了这里"插进去不重写"。

- **11 KV cache**（从 ch08 的 `KVSlice` 接缝换入）
  - 读 `llama.cpp/src/llama-kv-cache.h/.cpp`（`llama_kv_cache`，每层 k/v tensor `[n_embd_k_gqa, kv_size, …]`+ `llama_kv_cells` 簿记）。
  - 由 n_ctx → 算出 KV 尺寸并分配每层 k/v；把 ch08 `forward` 内部的「重算 append」换成「写/读真缓存」；mask 从 `[n_tokens×n_tokens]` 换成 `[n_kv×n_tokens]`。
  - 只做**连续单序列**场景（跳过 swap/shift/streaming）。
  - 产出：`Context`（含 KV 分配）+ 「写→读」回环测试。
- **12 batch / ubatch + 切分**
  - 读 `llama.cpp/src/llama-batch.h/.cpp`（`llama_batch`/`llama_ubatch`/`llama_batch_allocr`，`llama_batch_get_one`）。
  - 用户一次给一批 token（含 pos/seq_id），按 `n_ubatch` 切成物理子批。
  - 产出：batch 结构 + 单序列/多序列切分测试。

---

## ⑤ 前向算子深挖（13–22）——在 ch08 基础上逐算子深化

> 这一段是「一章一个算子/概念」的深挖主干，**建立在已跑通的 ch08 前向之上**。每章深化一个 ch08 刻意
> 精简/跳过的算子，必**新增** ch08 没有的内容（fused/布局注意力、RoPE 推导、F32 累加精度等），避免
> 沦为对 ch08 的重复推导。核心驱动：`llm_graph_context`（`llama-graph.h`）与 `build_graph`（`llama-model.cpp`）；
> 标准 llama 前向在 `src/models/llama.cpp` 的 `graph<false>`。

- **13 图驱动：graph context + build_graph**
  - 读 `llama.cpp/src/llama-graph.h`（`llm_graph_context` 的 build_* 帮手）+ `llama-model.cpp` 的 `build_graph`。
  - 定义「图构建环境」（派生维度 + 各类 input/reserve）与图产出结构。**这是 11+ 之后引入 batch 时才真正必要** —— ch08 用 3 句话点过此概念，本章落地。
  - 产出：图构建骨架（能登记各子模块产出的 tensor）+ 空图测试。
- **14 输入：token 嵌入 + 位置**
  - 读 `llama.cpp/src/llama-graph.cpp` `build_inp_embd` / `build_inp_pos`。
  - token 查 `tok_embd` 表得 hidden states；为每个位置生成 RoPE 位置。
  - 产出：嵌入查表 + 位置张量 + 测试。
- **15 归一化：RMSNorm（深化 ch08）**
  - 读 `llama-graph.cpp` `build_norm` + `llm_norm_type`（含 LayerNorm/GroupNorm，先只做 RMSNorm）。
  - 深化：RMS 的数值推导、eps 的作用、与 LayerNorm 对比。
  - 产出：RMSNorm 内核 + 数值测试（对照公式）。
- **16 QKV 投影 + 注意力头重塑（深化 ch08）**
  - 读 `llama-graph.cpp` `build_qkv` + 头 reshape/permute。
  - 深化：`permute(0,2,1,3)` 的布局语义（ch08 用可见循环模拟）。
  - 产出：QKV 构建 + 形状/索引测试。
- **17 RoPE 位置编码（深化 ch08）**
  - 读 `llama-graph.cpp` RoPE 应用 + 频率计算（rope_type / freq_base / scaling）。
  - 深化：旋转矩阵的数学推导、freq_base 意义。
  - 产出：RoPE 前向 + 频率表 + 数值测试。
- **18 注意力打分内核：KQ·Q → softmax → KQV·V → wo（深化 ch08）**
  - 读 `llama-graph.cpp` `build_attn_mha`。
  - 深化：fused 布局注意力 vs ch08 朴素逐位置循环；`ggml_mul_mat_set_prec(kq, F32)` 的 F32 累加精度动机。
  - 产出：完整单头注意力 + 与手算对照测试。
- **19 FFN：SILU + gate/up/down（深化 ch08）**
  - 读 `llama-graph.cpp` `build_ffn`。
  - `up` 线性 → SiLU → 与 `gate` 逐元素乘 → `down`（含量化 matmul）。
  - 产出：FFN + 数值测试。
- **20 输出：最终 norm + lm_head → logits（深化 ch08）**
  - 读 `src/models/llama.cpp` 末尾（final norm + `build_lora_mm(model.output,…)`）。
  - 汇总各层残差后的最终 RMSNorm + vocab 维投影得 logits。
  - 产出：整图输出 `t_logits` + 形状测试。
- **21 KV cache 接线入图（ch11 的图化）**
  - 读 `llama-graph.cpp` `build_attn_inp_kv`（k_idxs/v_idxs）+ `build_attn_inp_kq_mask` + `llama-kv-cache.cpp`（cpy_k/cpy_v、cells、prepare/update）。
  - 把「本轮该写缓存哪几列」与「因果 mask」织进图。
  - 产出：注意力输入接线 + mask/索引测试。
- **22 采样器进阶（optional 拆分）**
  - 读 `llama.cpp/src/llama-sampling.h/.cpp`（top-k / top-p / min-p / 重复惩罚 / grammar 后端）。
  - ch09 只有 greedy；本章补 top-k/top-p/惩罚。
  - 产出：多采样器 + 概率/复现测试。

---

## ⑥ 计算执行 + backend 抽象（23–26）——先 CPU，留 backend 槽位

这是用户关心的核心：**算子的「建图函数」与「后端内核」分离**。图是 backend 无关的；执行的本质是「遍历图 → 按 op 分发给 backend」。

- **23 计算图执行：遍历拓扑序并逐个算**
  - 读 `llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c` 的 `ggml_graph_compute` 遍历 + `ggml_compute_forward` 的 `switch(op)`。
  - 实现一个**最小执行器**：`compute_forward(node)` 从 `node->op` 分发给 `compute_forward_<op>`；布局 op（RESHAPE/VIEW/PERMUTE/CONT）纯元数据跳过。
  - 产出：图执行 + 用 13–22 造的图跑出 logits 并和人工抽查测试。
- **24 backend 抽象：接口 + 分发表**
  - 读 `llama.cpp/ggml/src/ggml-backend-impl.h` 的 `struct ggml_backend_i` + `ggml-backend.cpp` 的 `ggml_backend_graph_compute`。
  - 定义 `Backend` V 表：`get_name / free / graph_compute`（async/transfer 留 NULL 槽位）；实现 **CPU Backend** 的 `graph_compute`（就是 23 的遍历 → 分发表）。
  - 关键设计：**`op → 各 backend 的 compute 函数` 编成同一张分发表**，CPU 先填实，Metal（27–29）填同一张表的另一半。
  - 产出：`Backend` 接口 + `CPU::graph_compute` + 用统一入口跑通前向。
- **25 CPU 内核 · 逐元素/规约算子**
  - 对照 `ggml-cpu.c` 的 `ggml_compute_forward_*`：`add / mul / silu / rms_norm / soft_max / get_rows`。
  - 逐个手写 CPU 内核，填进 24 的分发表。
  - 产出：上述内核 + 每核数值测试。
- **26 CPU 内核 · mul_mat（最重算子）**
  - 对照 `ggml_compute_forward_mul_mat` + F16 数据路径；把 ch08 的 `matmul(float*)` 换成本章内核并对比。
  - 产出：F16/F32 `mul_mat` + 与手算小矩阵对照测试。

> 里程碑：**CPU 能跑出 logits → token。** server 可在 24 后接上并发/流式能力。

---

## ⑦ Metal backend（27–29）——第二后端

- **27 Metal backend 接入同一条分发表**
  - 对照 `llama.cpp/ggml/src/ggml-metal/ggml-metal.cpp`（`ggml_backend_metal_init`）+ `ggml-backend-impl.h`。
  - 用 Metal 填 `Backend` V 表（`graph_compute` 走 Metal 命令缓冲）；buffer 类型/内存分配。
  - 产出：`Metal::graph_compute` 骨架 + 至少一个内核能跑通（如 add），与 CPU 结果对比测试。
- **28 Metal 内核：读 .metal + 关键算子分派**
  - 对照 `ggml-metal-ops.cpp` 的大 `switch(op)` + `ggml-metal.metal` 里单个 kernel（如 silu/soft_max/rope）。
  - 实现 llama 前向必需的 kernel：mul_mat / soft_max / rope / rms_norm（挑最小集：add/silu 已覆盖，补 mul_mat + softmax + rope）。
  - 产出：Metal 跑通一次完整前向，logits 与 CPU 一致（数值容差测试）。
- **29 Metal 下的 server：同一套接口切换后端**
  - 在 ch10 的 server 里加一个「后端切换」，把 CPU 换成 Metal，其余逻辑不变。
  - 产出：`--backend cpu|metal` 都能跑的 server + 双端对比。

> 里程碑：**同一份 server、同一张分发表，CPU/Metal 都能推理。**

---

## ⑧ 收尾

- **30 汇总与展望**
  - 串讲全书分层（文件→语义→快路径前向/采样/server→上下文优化→算子深挖→执行/后端→server 切 Metal），总结「为什么 backend 抽象值得一开始留」。
  - 展望可后补的扩展：更多采样器（min-p/重复惩罚/grammar）、MoE、flash attention、量化推理、缓存复用等（这些上一阶段都标记为 EXT，留给想深入的人）。

---

## 篇幅与取舍

- **总 31 章（00–30）**。相比旧草案的调整：把「前向」拆成「快路径一章（08）+ 逐算子深挖若干章（13–22）」，让 server 里程碑提前、算子深挖建立在已跑通的前向之上。
- ⑤ 前向深挖占 10 章（13–22）是因为它真是「一个算子一章」——每章都能独立手写 + 数值测试，不偷工；且每章都**新增** ch08 刻意精简的内容，非重复推导。
- 采样器：greedy 在 **09** 先用起来；top-k/top-p/惩罚可并入 **22（optional）** 或「30 展望」。
- **量化类型体系（Q1_0/Q4_*…block 布局）暂不成章**（用户拍板：暂不补、留展望）。它是 llama.cpp / GGUF 的立身之本（体积缩小 3–4×），但因为 tinybrainbot 主线只用 F16/F32，为先把 CPU+Metal server 跑通、压住篇幅，先不单列。手头有 Bonsai Q1_0 对照文件，后续作为扩展补上最合适。mul_mat 内核（26 章）会给量化留一句钩子。
- 其余核心精髓（GQA 结构、RoPE yarn/动态缩放、KV cache shift、flash attention、`ggml_gallocr` 生命周期分配、采样器进阶）均归入 30 章展望——它们重要但不是跑通最小 CPU+Metal server 的必经之路。
- 每章默认「源码 + 手写测试」（非第三方框架、退出码非 0 = 失败），与 01–04 约定一致。

---

## 待你拍板

已拍板（2026-09）：
1. **server-first 快路径**：先搭 08（完整前向）→ 09（采样）→ 10（最小 server）拿 CPU server 出文本，KV cache / batch 延后为 11/12 优化章节。✅
2. **08 前向结构**：Oracle 裁定**方案 A（直写循环、不建图）**，但从第一天留 `KVSlice`/`build_causal_mask`/`forward` 三个接缝 + op 用裸 `float*`。✅
3. **逐算子深挖**：放 server 之后（13–22），每章深化一个 ch08 刻意精简的算子，新增 fused 注意力/RoPE 推导/F32 精度等内容。✅

仍待定：
4. **后端顺序**：先 CPU server 再 Metal（当前排期）——是否希望 **先 Metal 后 CPU server**（server 只做一次，后端是 CPU+Metal 二选一）？
5. **MVP server 的 HTTP 库**：cpp-httplib vs 手写极简 socket——**延后到 ch10 写时再定**。
6. **ch22 采样器进阶**：是否单拆（top-k/top-p/惩罚），还是只并入「30 展望」？
