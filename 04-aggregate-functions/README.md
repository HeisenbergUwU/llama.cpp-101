# 04 - 文件 IO 封装层（llama_file + llama_mmap）

> 把裸系统调用（`open`/`fstat`/`mmap`/`munmap`/`close`）封装成两个小类型，让上层逻辑不掺和 OS 细节。

## 做什么

前面 01 用裸 `FILE*`/`fread`，本章统一成**文件 IO 封装层**：

- **`llama_file`**：打开文件、`fstat` 拿大小、析构 `close`。
- **`llama_mmap`**：把整个文件 `PROT_READ | MAP_SHARED` 映射进地址空间、析构自动 `munmap`。

这是后续所有章（04+）唯一的文件 IO 入口——`gguf.cpp`、`llama-model.cpp` 不再裸调系统调用。`llama_mmap` 禁拷贝、可移动（mmap 是独占资源，靠移动语义 + 源 `addr` 置空防双 `munmap`）。

## 怎么跑

```bash
cd 04-aggregate-functions && make run
```

## 关键文件

| 文件 | 作用 |
|------|------|
| `include/llama-io.h` | `llama_file` + `llama_mmap` 声明（含移动语义） |
| `src/llama-io.cpp` | open/fstat/mmap/munmap/close 的真实实现 |
| `src/gguf.cpp` / `include/gguf.h` | 复用 `llama_file` 的 GGUF 解析（01 章的 IO 版） |
| `tests/test-llama-io.cpp` / `test-gguf-io.cpp` | 手写测试：mmap 基址读出 `GGUF` magic 验证零拷贝 |

## 对照上游

- `llama.cpp/src/llama-mmap.h` —— `struct llama_file`（`size()`/`file_id()`）与 `struct llama_mmap`（`size()`/`addr()`）
- `llama.cpp/src/llama-mmap.cpp` —— `llama_mmap::impl` 构造（`mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0)`）

> 教学取舍：去掉上游的 `prefetch`/`numa`/`MAP_POPULATE`/pimpl（那些是性能优化与封装技巧，非 mmap 原理）。更精简对照见 `reference.md`。

---

下一章：**05 - 建 `llama_model` 聚合对象 + 加载权重**（用 03 建 110 个 tensor，mmap 零拷贝挂权重）。
