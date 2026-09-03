# 10 章 · 参考源码对照

本章「把 llama 完整前向建成一张 ggml 计算图（整体模型图）」主要参考：

- `llama.cpp/src/llama-graph.cpp` —— `build_attn_mha`（注意力 = 组合：`ggml_permute(q/k/v, 0,2,1,3)`
  头重排 → `ggml_mul_mat(k,q)` 打分（`mul_mat_set_prec(F32)`）→ `ggml_soft_max_ext(kq, kq_mask,
  kq_scale)` → `ggml_mul_mat(v,kq)` → `ggml_permute`+`ggml_cont_2d` 重组头）；`build_attn_inp_kq_mask`
  （`[n_kv,n_tokens]`）；`build_inp_embd`（get_rows 查表）；`build_norm`（RMSNorm）；`build_ffn`
  （SwiGLU-PAR）；`build_lora_mm`（lm_head）
- `llama.cpp/src/models/llama.cpp` —— `graph<false>`（embed→12 层 attention+FFN→output_norm→tied
  lm_head，残差两处 ADD）
- `llama.cpp/ggml/include/ggml.h` / `ggml/src/ggml.c` —— `GGML_OP_*` 枚举、`ggml_new_tensor_impl`
  （view 共享 data）、`ggml_set/get_op_params_i32/f32`、`GGML_ASSERT`
- `llama.cpp/ggml/src/ggml-ops.c` —— `ggml_get_rows`、`ggml_soft_max_ext`、`ggml_permute`、
  `ggml_cont(_2d)`、`ggml_concat`、`ggml_view_*`
- `llama.cpp/ggml/src/ggml-cpu/*.c` —— `soft_max_f32`、`rms_norm_f32`、`rope_f32` 内核数值；
  `ggml_compute_forward_dup`（cont 按 nb 步长拷贝）

设计取舍（最小化结构版）：
- **无独立「注意力 kernel」**：用已有 `mul_mat` + `soft_max_ext` + `permute`/`cont` 在图上组合
  （对齐 `build_attn_mha`）；Q·Kᵀ 与 ·V 复用同一 `mul_mat`，F32 累加（本项目 `kernel::matmul`
  本已 F32 累加，满足 `mul_mat_set_prec(F32)` 精度动机）。
- **GQA 用「逐头 2D」**：每 kv 头服务 n_gqa 个 q 头，头切片 `head_extract`（strided 切列）、
  重组 `concat`（沿特征轴逐 token 交错）——不引入 4D/nb 泛化。
- **为跑 N>1 整图修的底层**（分发层，不动数值算法）：`mul_mat` 按 a 行循环 + 输出 `{n_out,R}`；
  `rms_norm` 按 token 行循环；`get_rows` 用 `dequant_row`（token_embd 是 F16，不能 memcpy 当 F32）；
  `rope` 逐行逐头 + positions 叶子。对齐 09 骨架「只做单行」与真实批量前向的落差。
- 掩码用 `-inf`（未来 token 不可见），经 `soft_max_ext` 指数前加入、被 softmax 归一为 0。
