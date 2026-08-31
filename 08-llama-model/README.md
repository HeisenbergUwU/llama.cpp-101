# 08 - llama_model 整合（进行中）

> 目标是让 `llama_model` 成为「带完整语义」的聚合对象；当前章节仍处在搭建/规划阶段。

## 当前状态

目录是 07 的延续起步（内容暂与 07 相同：`llama_model` 已集成 `HParams hparams` 与 `Vocab vocab`，`load_model` 在加载时一并完成 hparams 解析与词表构建）。本章作为**独立章节**，方向是把 `llama_model` 继续完善成后续前向计算真正依赖的载体。

> ⚠️ 目前 `src/`、`include/`、`tests/` 仍是 07 词表那套代码的拷贝，尚未写入本章独有的实现。

## 规划方向（对应 ROADMAP ④ 上下文层）

- `llama_context` 初始化 + KV 缓存分配（`llama.cpp/src/llama-context.*`、`llama-kv-cache.*`）
- 由 `n_ctx` 算出每层 k/v 尺寸并分配 + `llama_kv_cells` 簿记

> 具体拆分以后续开发为准，本章内容随实现逐步完善。

## 怎么跑（当前暂与 07 相同）

```bash
cd 08-llama-model && make run
```

如果后续补上了 context/KV 的实现，本章「做什么」「怎么跑」「关键文件」会随之更新。

---

上一章：**07 - 词表 Vocab（tokenize / detokenize）**。
