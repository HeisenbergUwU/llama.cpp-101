# 02 - 迷你 ggml 数据结构层（ggml_context + ggml_tensor）

> 用一块连续内存（池子）装一批底层张量结构及其绑定关系，立起 `ggml_context` / `ggml_tensor` / 类型的骨架。**只建结构，不写任何函数**。

## 做什么
- 定义内存池 `ggml_context`，整个池子是一块 calloc 的连续内存（`mem_size` / `mem_buffer`）。
- 用 `ggml_object` 链表串联池子里的对象（`offs` / `size` / `next` / type），实现「对象→池子」绑定。
- `ggml_tensor` 直接切在池子内（`data` 指向数据）；`no_alloc` 支持 mmap 零拷贝挂权重。
- 定义 `ggml_type`：只列 F32(0) / F16(1) / Q1_0(41)，编号照抄 GGUF 契约。
- 定义张量形状 `ne[]`/`nb[]`（行主序，`nb[i]=nb[i-1]×ne[i-1]`）与 `nbytes` 公式。
- 声明 `ggml_cgraph`（nodes / leafs / use_counts / eval_order），只描述供 03 建图，不实现。

> 范围：不建图、不执行（`ggml_cgraph` 只描述）；也不把 GGUF 读成 tensor（那是后续 `llama_model`）。池子分配函数放到 03 章。

## 怎么跑

本章无 Makefile、不可独立构建（只写类型与布局，无函数实现）。

## 关键文件
| 文件 | 作用 |
|------|------|
| `include/ggml.h` | 类型层声明：`ggml_context` / `ggml_tensor` / `ggml_type` / `ggml_init_params` / `ggml_cgraph` |
| `src/ggml.cpp` | 私有定义 `ggml_object` / `ggml_context` 内部布局（绑定只握在拥有者手里） |
| `test/` | 空目录（无测试、无 Makefile，本章不构建） |
| `reference.md` | 参考源码对照 |

## 对照上游
- `llama.cpp/ggml/include/ggml.h` —— `ggml_tensor` / `ggml_context` / `ggml_init_params` / `ggml_type` / `ggml_cgraph` 声明
- `llama.cpp/ggml/src/ggml.c` —— `ggml_object` / `ggml_context` 内部布局（`offs`/`size`/`next`、`objects_begin/end`、`no_alloc`）
- `llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c` —— F16 / Q1_0 的 `to_float` 解量化（`ggml_table_f32_f16`）

---
下一章：**03 - 迷你 ggml 加载层**（在 02 的结构上把 `ggml_init` / `ggml_new_tensor_*` 等池子分配函数真正写出来）。
