# 01 - 加载并校验 GGUF（load + check）

GGUF（**G**GML **U**nified **F**ormat）是 llama.cpp 存储和加载模型权重的二进制格式。它把**元信息（metadata）**和**张量权重（tensor data）**塞进同一个文件，加载方顺序读一遍就能还原模型，不需要额外索引文件。

> 范围：本文只讲 GGUF 的**文件格式**（怎么布局、怎么读）。张量内部的量化布局属于 ggml 的量化格式，不在本章范围。

## 一、文件总体结构

`gguf.h` 开头注释（`llama.cpp/ggml/include/gguf.h`）给出官方结构：

```
+---------------------------+
| 1. magic         "GGUF"   |  4 bytes
+---------------------------+
| 2. version        uint32  |  4 bytes
+---------------------------+
| 3. n_tensors      int64   |  张量个数
+---------------------------+
| 4. n_kv           int64   |  键值对个数
+---------------------------+
| 5. KV pairs       ...     |  n_kv 个元信息
+---------------------------+
| 6. tensor info    ...     |  n_tensors 个张量描述
+---------------------------+
|    （对齐的填充）          |
+---------------------------+
| 7. tensor data    blob    |  实际权重二进制
+---------------------------+
```

各字段：

- **magic（4B）**：ASCII `"GGUF"`，识别文件类型。
- **version（u32）**：格式版本，当前 `3`；读到更大版本直接报错。
- **n_tensors（i64）/ n_kv（i64）**：个数。
- **KV pairs**：n_kv 个元信息（架构、上下文、分词器、模板），规则见第三节。
- **tensor info**：n_tensors 个张量「描述」——**只有描述，没有数据**，权重在最后。
- **tensor data**：所有权重拼成的一个大 blob，起点按 `alignment` 对齐。

> 所有整数默认**小端序**。源码读取时做端序检测：把大端当小端读，版本号会变成一个极大的数（`gguf.cpp` 里 `version & 0x0000FFFF == 0` 就是提示端序不匹配）。

## 二、KV pairs 的排列规则

每个 KV 项在文件里的字节排列：

```
每个 KV 项：
  key   : [u64 长度][len 个字节]        —— 键名，不含 '\0'
  type  : int32 (gguf_type 枚举)        —— 值的类型
  value : 按 type 分三种：
          · STRING(8) : [u64 长度][len 个字节]
          · ARRAY(9)  : [int32 元素类型][u64 元素个数] + 连续 n 个元素
          · 标量       : 直接 n 字节原始数据（见下方“标量大小”表）
```

三种通用序列化约定：

- **字符串**：`[u64 len][len 字节]`，**不含结尾 `\0`**。
- **枚举**：一律存成 `int32`。
- **bool**：一律存成 `int8`。

### 标量大小表

| 类型                | 字节 | 类型                     | 字节 |
| ------------------- | :--: | ------------------------ | :--: |
| UINT8 / INT8 / BOOL |  1   | UINT32 / INT32 / FLOAT32 |  4   |
| UINT16 / INT16      |  2   | UINT64 / INT64 / FLOAT64 |  8   |

### 数组（ARRAY）多一层的特殊性

ARRAY 值比普通值多两层：先读**元素类型**（`int32`），再读**元素个数**（`uint64`），然后是逐个元素。例如「字符串数组」= `[u64 长度]` 后接连续 `[u64 长度 + 字节]` 的字符串列表。

> 别混淆两层类型：外层 `type` 才是 `GGUF_TYPE_ARRAY`(9)，内层是**数组元素类型**。`gguf.cpp` 先 `read(type)` 判 ARRAY，再 `read(type)` 覆盖为元素类型、`read(n)`。

例：`tokenizer.ggml.tokens` = `元素类型 STRING(8)` + `个数 32000` + 连续 32000 个字符串（词表）。

### KV 值类型全集

| 枚举    | 值  | 枚举    | 值  |
| ------- | :-: | ------- | :-: |
| UINT8   |  0  | BOOL    |  7  |
| INT8    |  1  | STRING  |  8  |
| UINT16  |  2  | ARRAY   |  9  |
| INT16   |  3  | UINT64  | 10  |
| UINT32  |  4  | INT64   | 11  |
| INT32   |  5  | FLOAT64 | 12  |
| FLOAT32 |  6  |         |     |

## 三、tensor info 的排列规则

KV 段之后是张量信息段，每个 tensor info 的字节排列：

```
每个 tensor info：
  name  : [u64 长度][len 个字节]     —— 张量名，如 "token_embd.weight"
  n_dims: uint32                     —— 维度数（≤ GGML_MAX_DIMS=4，通常 4）
  ne[]  : 每维 int64，共 n_dims 个    —— 每维大小；未填到的维度默认 1
  type  : int32 (ggml_type 枚举)      —— F32/F16/Q4_K 等
  offset: uint64                      —— 在末尾 tensor data blob 里的字节偏移
```

要点：

- 顺序是 ggml 的**行主序**：`ne[0]` 是最内层/每行元素数。
- 张量名不能超长（`GGML_MAX_NAME`）、不能重复。
- 字节大小由 `type` 和 `ne[]` 现算：`nbytes = (总元素数 / blck_size) × type_size`。对 F32/F16，`blck_size=1`，故 `nbytes = 总元素数 × 每元素字节数`。
- **张量信息只有描述**：权重全部集中放末尾 blob，每个张量靠自己的 `offset` 定位。这正是"先读元信息、再按需 mmap 权重"的关键。

## 四、数据对齐与 blob

读完全部 tensor info 后、读 blob 前做一次对齐（`gguf.cpp` 756 行）：

```cpp
if (n_tensors > 0 && !gr.seek(GGML_PAD(gr.tell(), ctx->alignment))) { ... }
```

即把当前位置**向上取整到 `alignment`（默认 32，或 KV 里 `general.alignment` 覆盖）的倍数**，从这里才是张量数据。

对齐公式 `pad(x,a)=(x+a-1)/a*a`：先加 `a-1` 把余数"顶"过一档再整除，就只进不舍，最后 `*a` 还原成字节偏移。

> 对齐作用：让权重在内存/mmap 里按 32 字节对齐，便于 SIMD 与向量化读取，避免跨 cache line。llama.cpp 上游 `GGML_PAD` 用位运算版（要求 `n` 是 2 的幂），故 GGUF 读取时把 `alignment` 校验为 2 的幂。

## 五、llama.cpp 源码引用

本章排列规则与校验全部对照 `llama.cpp/` 内实际源码，非凭记忆。解析流程看 `gguf.cpp` 的 `gguf_init_from_file`，bounds 校验看 `llama-model-loader.h` 的 `llama_tensor_weight`，词表（tokenizer）看 `llama-vocab.cpp` 的 `llama_vocab::load`。

| 相对 `llama.cpp/` | 用途 | 关键位置 |
|------|------|------|
| `ggml/include/gguf.h` | GGUF 结构定义 + 读取 C API | 头部注释（结构）、`gguf_type` 枚举 |
| `ggml/src/gguf.cpp` | `gguf_init_from_file` 解析实现 | KV / tensor / 对齐 / 校验 |
| `ggml/include/ggml.h` | `ggml_type` 枚举、`GGML_MAX_DIMS`、`GGML_MAX_NAME` | `ggml_type` 定义 |
| `src/llama-model-loader.h` | bounds 校验 `offs + nbytes <= file_size` | `llama_tensor_weight` |
| `src/llama-vocab.cpp` | **如何使用 tokens/scores/token_type 三个数组构建词表** | `llama_vocab::load` |

这三条词表 KV（`tokenizer.ggml.tokens/scores/token_type`)在 `llama-arch.cpp` 注册为 `LLM_KV_TOKENIZER_LIST / TOKENIZER_SCORES / TOKENIZER_TOKEN_TYPE`，`llama-vocab.cpp` 里 `gguf_find_key` 找到后：`tokens` 必读（缺失即拒载，建文字↔id 双向映射）、`scores` 可选（填 `token_data.score`，供 bpe 打分、`-1000` 低分使特殊 token 不被合并）、`token_type` 可选（映射成 `LLAMA_TOKEN_ATTR_*`，决定 token 在编码/解码/采样的行为）。这正是 02 章（llama_model）要把原始字节组织成语义词表对象的地方（02 目前只把权重建成 `ggml_tensor`，词表属于 03 模型语义阶段）。
