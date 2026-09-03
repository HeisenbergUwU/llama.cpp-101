# 11 章 · 参考源码对照

本章「自回归采样 + 跑起来（一个文字一个文字地推理）」主要参考：

- `llama.cpp/src/llama-sampler.h/.cpp` —— `llama_sampler` 接口；`llama_sampler_init_greedy`
  的 greedy 分支（`apply` = 在一批候选里挑 logit 最大的 token）；`llama_token_data` /
  `llama_sampler_sample` 采样收口
- `llama.cpp/src/llama.cpp` —— `llama_decode`（喂 token 拿 logits）、`llama_generate`
  （解码循环：decode → sample → 判 EOG → 续/停）
- `llama.cpp/src/llama-vocab.h/.cpp` —— `llama_vocab::is_eog`（句末判断；本迷你版 `Vocab::is_eog`）
- `llama.cpp/ggml/include/ggml.h` / `ggml/src/ggml.c` —— `ggml_used_mem`（池子已用字节，
  本章新增迷你版 `ggml_mem_used`）

> 10 章的整体模型图（`build_model_graph`）对照清单见 `10-llama-model-graph/reference.md`；
> 本章在其上复用，另补「采样器 + 解码循环」这一层。
