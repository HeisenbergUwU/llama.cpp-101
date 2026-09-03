# 09 章 · 参考源码对照

本章「ggml 计算图设施」主要参考：

- `llama.cpp/ggml/include/ggml.h` —— `ggml_cgraph`（nodes/leafs/use_counts/visited_hash_set）、
  `ggml_build_forward_expand` / `ggml_build_forward_impl`、`ggml_graph_compute` 声明
- `llama.cpp/ggml/src/ggml.c` —— `ggml_new_graph(_custom)`（建空图）、`ggml_build_forward_impl` +
  `ggml_visit_parents_graph`（递归登记依赖、去重、分叶子/节点）、`ggml_graph_compute`
- `llama.cpp/ggml/src/ggml-alloc.c` —— `ggml_gallocr_alloc_graph`（按 use_counts 算生命周期、
  复用 arena 缓冲）—— 本课件为「每节点独立 data」简化版，未实现内存复用（留作后续章）
- `llama.cpp/ggml/src/ggml-cpu/*.c` —— 算子数值实现；本项目算子数值用 08 章 kernel
  （namespace kernel）

做法（最小化结构版）：
- 图设施独立在 `namespace ggml`（ggml_graph / ggml_node），不入 llama 层、不接模型。
- 权重是图里的「叶子节点」：指向加载层 mmap 的 `ggml_tensor`（零拷贝、只读），对齐上游。
- build 用后序 + visited 去重（共享节点只登记一次）；compute 按拓扑 dispatch 到 kernel。
