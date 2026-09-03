# 04 章 · 参考源码对照

本章「文件 IO 封装层」主要参考：

- `llama.cpp/src/llama-mmap.h` —— `struct llama_file`（文件封装，`size()`/`file_id()`）与 `struct llama_mmap`（mmap 映射，`size()`/`addr()`）
- `llama.cpp/src/llama-mmap.cpp` —— `llama_mmap::impl` 构造（`mmap(NULL, file->size(), PROT_READ, MAP_SHARED, fd, 0)`）、`llama_file::impl`（open + fstat）

做法（最小化结构版）：
- 04 章 = 把裸系统调用（`open`/`fstat`/`mmap`/`munmap`/`close`）封装成 `llama_file` + `llama_mmap` 两个小类型：`llama_file` 打开文件、`fstat` 拿大小、析构 `close`；`llama_mmap` 把整文件 `PROT_READ | MAP_SHARED` 映射进地址空间、析构 `munmap`。
- 教程去掉上游的 `prefetch`/`numa`/`MAP_POPULATE`/`posix_madvise`/pimpl（性能优化与封装技巧，非 mmap 原理）。
- 04 章只交付 IO 封装 + 独立测试（`mmap` 基址读出 `GGUF` magic 验证零拷贝）；用它撮合 gguf/ggml/权重是 05 章的事。
