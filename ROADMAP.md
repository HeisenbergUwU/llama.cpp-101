# llama.cpp-101 路线图（草案 · 20–30 章）

> **状态：草案，待审阅**。目标是把整条链路拆到「CPU + Metal 的 server 推理」。
> 每一章对应一个**可独立讲、可手写代码 + 手写测试**的概念，且**有真实上游源码可对照**。
> 用 tinybrainbot（plain llama，F16/F32）当测试模型——它不涉及 MoE/SWA/flash-attn，正好能走通一条最小的「必须讲」主线。
>
> 上游文件路径以 `llama.cpp-101/llama.cpp/` 为根（相对路径）。文件名/函数已按当前 clone 核对；为稳健起见，文档只精确到「概念 ↔ 文件 ↔ 关键函数」，不绑易漂移的行号。

---

## 🎯 精简主线（当前优先级：先做到「CPU server 能 chat completion 一个小模型」）

> 下方的大路线图是长期参考；**这里才是最先要走的快路径**。基于 tinybrainbot 实测
> （GQA、无 bias、无独立 lm_head 复用 token_embd、SwiGLU-PAR、RMS eps=1e-5、
> RoPE linear base=10000/rot=64），把「CPU server 出文本」压缩到最少章节：
>
> MVP 先**全量重算、不引入 KV cache**（每次喂全部历史 token 重算前向），先把链路跑通；
> KV cache 是后续优化章节。

| 步 | 内容 | 产出 |
|----|------|------|
| **03** | 模型语义 + 单 token 前向：从 `gguf_context.kv` 解析 hparams/vocab（tokenize/detokenize）；写 llama 前向 13 步算子（复用 `model.require_weight`）；argmax 采 1 个 token | 给一个 prompt 能算出 logits + 贪心 1 token |
| **04** | 自回归采样：贪心循环生成多 token，直到 EOG / max_tokens | 命令行能生成一段文本 |
| **05** | 最小 CPU server：HTTP + `POST /v1/chat/completions`（非流式先出 JSON） | **curl 一发，拿到回复文本** |
| 06（延后） | KV cache 优化 + 流式 SSE + Metal 第二后端 | 见下方大路线图 |

> 先打卡「chat completion 出文本」这个里程碑，其余（KV cache / quant / Metal / 批处理）
> 作为后续优化章节再接。

---

## 阶段总览（完整路线，长期参考）

| 阶段 | 章节 | 主题 | 里程碑 |
|------|------|------|--------|
| ①② 基础 | 00–02 | 什么是 / GGUF 解析 / 权重加载 | ✅ 已完成 |
| ③ 模型语义层 | 03–05 | hparams → llama_model → vocab | 拿到「语义模型」 |
| ④ 上下文层 | 06–07 | context + KV 分配 / batch | 拿到「运行内存」 |
| ⑤ 前向计算图 | 08–17 | transformer 逐算子拆解 | 图能描述一次前向 |
| ⑥ 计算执行 + backend | 18–21 | 图执行 / backend 抽象 / CPU 内核 | **CPU 能跑出 logits → token** |
| ⑦ CPU server MVP | 22–24 | HTTP / 批处理 / 流式 | **CPU server 出文本** |
| ⑧ Metal backend | 25–27 | 接同一条分发表 / 内核 / server 切 Metal | **CPU+Metal 双后端 server** |
| ⑨ 收尾 | 28 | 汇总结论 | 全书收官 |

> 编排说明：**先让 CPU 的 server 能吐出文本**（22–24），再把 Metal 作为**第二后端插进同一条分发表**（25–27）。
> 这样每一步都有可运行的产出；而 Metal 正好演示「为什么当初要留 backend 槽位」。

---

## ①② 基础（已完成）

- **00** `00-what-is-llama-cpp/` — 什么是 llama.cpp。✅
- **01** `01-load-and-check-gguf/` — 裸 GGUF 解析器（header / KV / tensor info / 对齐 / bounds）。✅
- **02** `02-model-loader/` — 建聚合对象 `llama_model`（迷你 ggml + 110 个真实 `ggml_tensor` 池子式内存 + mmap 零拷贝）。✅
  - 关键：把权重「拿到内存」**不需要任何计算算子**（纯 memcpy/指针），算子是 18–21 的后话。

---

## ③ 模型语义层（03–05）

从 02 的「扁平张量表」升级成「有语义的模型」。

- **03 hparams：哪些 KV 决定架构**
  - 读 `llama.cpp/src/llama-hparams.h` `struct llama_hparams` + `llama-hparams.cpp` 的 `n_head()/n_layer()/n_embd_head_k…` 派生方法。
  - 从 GGUF KV 读出 n_embd、n_layer、n_head(_kv)、n_ff、rope 参数、norm eps…并算派生维度。
  - 产出：`HParams` + 解析 + 核对 tinybrainbot 实测值 + 测试。
- **04 llama_model：tensor 组装成语义层**
  - 读 `llama.cpp/src/llama-model.h`（`llama_model` + `llama_layer`）+ `llama-model.cpp` 的 `create_tensor`。
  - 把 02 的张量挂到「每层一个 bag（attn_norm/wq/wk/wv/wo/ffn_norm/gate/up/down）」+ 根张量（tok_embd/output_norm/output）。
  - 产出：`Model` 语义对象 + 层数/每层张量完备性测试。
- **05 vocab + 分词/去分词**
  - 读 `llama.cpp/src/llama-vocab.h/.cpp`（`llama_vocab`，`llama_tokenize`/`llama_token_to_piece`/`llama_vocab_is_eog`）。
  - 从 `tokenizer.ggml.*` KV 原始字节建词表，实现 tokenize（文字→id）与 detokenize（id→文字）。
  - 产出：`Vocab` + round-trip 测试（token↔text）。

---

## ④ 上下文层（06–07）

运行态内存与「一次喂多少 token」。

- **06 context 初始化 + KV 缓存分配**
  - 读 `llama.cpp/src/llama-context.h/.cpp`（`llama_context`，`llama_init_from_model`）+ `llama-cparams.h`（n_ctx/n_batch/n_ubatch）+ `llama-kv-cache.h/.cpp`（`llama_kv_cache`，每层 k/v tensor `[n_embd_k_gqa, kv_size, …]`）。
  - 由 n_ctx → 算出 KV 尺寸并分配每层 k/v + `llama_kv_cells` 簿记。
  - 产出：`Context`（含 KV 分配）+ 尺寸核对测试。
- **07 batch / ubatch + 切分**
  - 读 `llama.cpp/src/llama-batch.h/.cpp`（`llama_batch`/`llama_ubatch`/`llama_batch_allocr`，`llama_batch_get_one`）。
  - 用户一次给一批 token（含 pos/seq_id），按 `n_ubatch` 切成物理子批。
  - 产出：batch 结构 + 单序列/多序列切分测试。

---

## ⑤ 前向计算图（08–17）——transformer 逐算子

这一段是「一章一个算子/概念」的主干。核心驱动：`llm_graph_context`（`llama-graph.h`）与 `build_graph`（`llama-model.cpp`）；标准 llama 前向在 `src/models/llama.cpp` 的 `graph<false>`。

- **08 图驱动：graph context + build_graph**
  - 读 `llama.cpp/src/llama-graph.h`（`llm_graph_context` 的 build_* 帮手）+ `llama-model.cpp` 的 `build_graph`。
  - 定义「图构建环境」（派生维度 + 各类 input/reserve）与图产出结构。
  - 产出：图构建骨架（能登记各子模块产出的 tensor）+ 空图测试。
- **09 输入：token 嵌入 + 位置**
  - 读 `llama.cpp/src/llama-graph.cpp` `build_inp_embd` / `build_inp_pos`。
  - token 查 `tok_embd` 表得 hidden states；为每个位置生成 RoPE 位置。
  - 产出：嵌入查表 + 位置张量 + 测试。
- **10 归一化：RMSNorm**
  - 读 `llama-graph.cpp` `build_norm` + `llm_norm_type`（含 LayerNorm/GroupNorm，先只做 RMSNorm）。
  - 产出：RMSNorm 内核 + 数值测试（对照公式）。
- **11 QKV 投影 + 注意力头重塑**
  - 读 `llama-graph.cpp` `build_qkv` + 头 reshape/permute。
  - Q/K/V 各自投影并切成 `[n_head, head_dim]` 布局。
  - 产出：QKV 构建 + 形状/索引测试。
- **12 RoPE 位置编码**
  - 读 `llama-graph.cpp` RoPE 应用 + 频率计算（rope_type / freq_base / scaling）。
  - 产出：RoPE 前向 + 频率表 + 数值测试。
- **13 注意力与 KV cache 接线**
  - 读 `llama-graph.cpp` `build_attn` + `build_attn_inp_kv`（k_idxs/v_idxs）+ `build_attn_inp_kq_mask`。
  - 把「本轮该写缓存哪几列」与「因果 mask」织进图。
  - 产出：注意力输入接线 + mask/索引测试。
- **14 注意力打分内核：KQ·Q → softmax → KQV·V → wo**
  - 读 `llama-graph.cpp` `build_attn_mha`。
  - 缩放点积、因果 mask、softmax、再乘 V、投影回 wo。
  - 产出：完整单头注意力 + 与手算对照测试。
- **15 KV cache 存取**
  - 读 `llama.cpp/src/llama-kv-cache.cpp`（cpy_k/cpy_v、cells、prepare/update）。
  - 把本轮 k/v 写进缓存列、从缓存列取历史。
  - 产出：KV 读写 + 「写→读」回环测试。
- **16 FFN：SILU + gate/up/down**
  - 读 `llama-graph.cpp` `build_ffn`。
  - `up` 线性 → SiLU → 与 `gate` 逐元素乘 → `down`（含量化 matmul）。
  - 产出：FFN + 数值测试。
- **17 输出：最终 norm + lm_head → logits**
  - 读 `src/models/llama.cpp` 末尾（final norm + `build_lora_mm(model.output,…)`）。
  - 汇总各层残差后的最终 RMSNorm + vocab 维投影得 logits。
  - 产出：整图输出 `t_logits` + 形状测试。

---

## ⑥ 计算执行 + backend 抽象（18–21）——先 CPU，留 backend 槽位

这是用户关心的核心：**算子的「建图函数」与「后端内核」分离**。图是 backend 无关的；执行的本质是「遍历图 → 按 op 分发给 backend」。

- **18 计算图执行：遍历拓扑序并逐个算**
  - 读 `llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c` 的 `ggml_graph_compute` 遍历 + `ggml_compute_forward` 的 `switch(op)`。
  - 实现一个**最小执行器**：`compute_forward(node)` 从 `node->op` 分发给 `compute_forward_<op>`；布局 op（RESHAPE/VIEW/PERMUTE/CONT）纯元数据跳过。
  - 产出：图执行 + 用 08–17 造的图跑出 logits 并和人工抽查测试。
- **19 backend 抽象：接口 + 分发表**
  - 读 `llama.cpp/ggml/src/ggml-backend-impl.h` 的 `struct ggml_backend_i` + `ggml-backend.cpp` 的 `ggml_backend_graph_compute`。
  - 定义 `Backend` V 表：`get_name / free / graph_compute`（async/transfer 留 NULL 槽位）；实现 **CPU Backend** 的 `graph_compute`（就是 18 的遍历 → 分发表）。
  - 关键设计：**`op → 各 backend 的 compute 函数` 编成同一张分发表**，CPU 先填实，Metal（25–27）填同一张表的另一半。
  - 产出：`Backend` 接口 + `CPU::graph_compute` + 用统一入口跑通前向。
- **20 CPU 内核 · 逐元素/规约算子**
  - 对照 `ggml-cpu.c` 的 `ggml_compute_forward_*`：`add / mul / silu / rms_norm / soft_max / get_rows`。
  - 逐个手写 CPU 内核，填进 19 的分发表。
  - 产出：上述内核 + 每核数值测试。
- **21 CPU 内核 · mul_mat（最重算子）**
  - 对照 `ggml_compute_forward_mul_mat` + F16 数据路径；可顺带讲 F32 对照与量化为何难。
  - 产出：F16/F32 `mul_mat` + 与手算小矩阵对照测试。

>CPU server 能并发、能流式吐出文本。**

---

## ⑧ Metal backend（25–27）——第二后端

- **25 Metal backend 接入同一条分发表**
  - 对照 `llama.cpp/ggml/src/ggml-metal/ggml-metal.cpp`（`ggml_backend_metal_init`）+ `ggml-backend-impl.h`。
  - 用 Metal 填 `Backend` V 表（`graph_compute` 走 Metal 命令缓冲）；buffer 类型/内存分配。
  - 产出：`Metal::graph_compute` 骨架 + 至少一个内核能跑通（如 add），与 CPU 结果对比测试。
- **26 Metal 内核：读 .metal + 关键算子分派**
  - 对照 `ggml-metal-ops.cpp` 的大 `switch(op)` + `ggml-metal.metal` 里单个 kernel（如 silu/soft_max/rope）。
  - 实现 llama 前向必需的 kernel：mul_mat / soft_max / rope / rms_norm（挑最小集：add/silu 已覆盖，补 mul_mat + softmax + rope）。
  - 产出：Metal 跑通一次完整前向，logits 与 CPU 一致（数值容差测试）。
- **27 Metal 下的 server：同一套接口切换后端**
  - 在 22–24 的 server 里加一个「后端切换」，把 CPU 换成 Metal，其余逻辑不变。
  - 产出：`--backend cpu|metal` 都能跑的 server + 双端对比。

> 里程碑：**同一份 server、同一张分发表，CPU/Metal 都能推理。**

---

## ⑨ 收尾

- **28 汇总与展望**
  - 串讲全书分层（文件→语义→上下文→图→执行→后端→server），总结「为什么 backend 抽象值得一开始留」。
  - 展望可后补的扩展：更多采样器（top-p/min-p/重复惩罚/grammar）、MoE、flash attention、量化推理、缓存复用等（这些上一阶段都标记为 EXT，留给想深入的人）。

---

## 篇幅与取舍

- **总 29 章（00–28）**，落在 20–30 区间，留一个数余量可增可减。
- ⑤ 前向占 10 章（08–17）是因为它真是「一个算子一章」——每章都能独立手写 + 数值测试，不偷工。
- 采样器（greedy 之外）**没有单列章**：greedy 放进 21 的里程碑里先用起来；top-k/top-p/惩罚等并入「28 展望」或用户后续想展开时再拆——避免把 20–30 撑爆。
- **量化类型体系（Q1_0/Q4_*…block 布局）暂不成章**（用户拍板：暂不补、留展望）。它是 llama.cpp / GGUF 的立身之本（体积缩小 3–4×），但因为 tinybrainbot 主线只用 F16/F32，为先把 CPU+Metal server 跑通、压住篇幅，先不单列。手头有 Bonsai Q1_0 对照文件，后续作为扩展补上最合适。mul_mat 内核（21 章）会给量化留一句钩子。
- 其余核心精髓（GQA 结构、RoPE yarn/动态缩放、KV cache shift、flash attention、`ggml_gallocr` 生命周期分配、采样器进阶）均归入 28 章展望——它们重要但不是跑通最小 CPU+Metal server 的必经之路。
- 每章默认「源码 + 手写测试」（非第三方框架、退出码非 0 = 失败），与 01/02 约定一致。

---

## 待你拍板

1. **章节粒度**：05 前向 10 章（08–17）是否偏细？可合并（如 10 归并到 11、15 并入 13/14）压到 6–7 章。
2. **后端顺序**：先 CPU server（22–24）再 Metal（25–27）——你是否希望 **先 Metal 后 CPU server**（server 只做一次，后端是 CPU+Metal 二选一）？
3. **采样器**：greedy 起步、其余并入展望，还是想单拆几章（top-k/top-p/惩罚）？
4. **MVP server** 的 HTTP 库：教程跟上游用 cpp-httplib，还是手写一个极简 socket（更「从零」但代码多）？
