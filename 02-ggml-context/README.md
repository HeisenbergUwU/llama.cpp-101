# 02 - 迷你 ggml：数据结构与内存池绑定（ggml_context + ggml_tensor）

01 章读懂了 GGUF 文件长什么样（一份顺序布局的字节流：元信息 + tensor 描述 + 权重 blob），但读到的只是一堆散装字节和一个 `offset`。要变成能用的模型，得有东西把这些描述**装进内存、组织成带形状的 `ggml_tensor`**，并让每个 tensor 的 `data` **绑定**到权重字节上。这层就是 ggml。

本章做**迷你 ggml**：用一块连续内存（池子）装一批底层张量结构 + 它们之间的绑定关系，把 `ggml_context` / `ggml_tensor` 的骨架立起来。

> 范围：只做**数据结构 + 池子分配**，建一个 context、往池子里塞 tensor。**不建图、不执行**（`ggml_cgraph` 只声明描述，真正建图在 03）；也不把 GGUF 读成 tensor（那是 03 `llama_model`）。

## 一、为什么要多一层 context？—— 池子

01 章结束时，你手里有 110 个 tensor 的**描述**（name/ne[]/type/offset）和一个 200MB 权重 blob。要做的事：把分散在文件上的 tensor 变成程序里**能按名索引的独立对象**，还要统一管内存和对齐。

朴素方案是 `malloc` 110 个 `ggml_tensor`。ggml 不用这个，它用一个**内存池（arena）**——**一个 `ggml_context` 就是一块"想装很多对象"的连续内存**，所有 tensor 都切在里面，不逐个 `malloc`：

```
                ggml_context（池子）
┌──────────────────────────────────────────────┐
│ mem_size   = 池子总字节数                      │
│ mem_buffer = 池子内存起点（calloc 的一块连续内存）│
│ objects_begin ─┐                               │
│ objects_end   ─┼─► ggml_object[0] ─► [1] ─►… 串联链表
└────────────────┴──────────────────────────────┘
```

## 二、context 的"绑定关系"（本章核心）

"绑定"有两层：**对象归属（谁住哪个池子）** 和 **数据指针绑定（data 指向哪块内存）**。

### 2.1 对象 → 池子：ggml_object 链表

`ggml_context` 内部不存数组，而存一条 `ggml_object` 链表（`ggml.cpp` 私有结构）：

```cpp
struct ggml_object {
    size_t offs;               // 对象在 mem_buffer 里的偏移
    size_t size;               // 占的字节数（含对齐）
    struct ggml_object *next;  // 把池子里的对象串起来
    enum ggml_object_type type;// TENSOR / GRAPH / WORK_BUFFER
};
```

每新建一个 tensor，就在池子末尾切一块 → 包成 `ggml_object` → `objects_end->next` 连上 → 更新 `objects_end`。

**绑定 A（对象→池子）**：`offs` 把对象钉在 `mem_buffer` 上，对象地址 = `mem_buffer + offs`。

### 2.2 tensor → context：tensor 直接放池子里

`ggml_tensor` 是 TENSOR 类型的 object 真正存的结构。切出一块后：

```cpp
struct ggml_tensor *t = (struct ggml_tensor *)((char*)ctx->mem_buffer + obj->offs);
```

**绑定 B（tensor→object→context）**：tensor 直接放在池子内存里，**只要知道 `ggml_context*`，遍历 `objects_begin` 链表就能走到它的每个 tensor**。这是 per-context 归属绑定。

### 2.3 data → 数据（零拷贝）

`ggml_tensor` 有 `data` 指针指向真实数据，位置分两种：

```
方式① 池子内分配（no_alloc=false）：obj 后面紧跟就放数据
    ┌──────────────┬──────────────┐
    │ ggml_tensor  │ 数据 bytes   │   ← 数据也住池子里
    └──────────────┴──────────────┘

方式② mmap 零拷贝（no_alloc=true，本项目用）：
    tensor 的结构在池子里，data 却【直接指向 mmap 映射的 GGUF 权重区】
    data = mmap_base + offset
```

**绑定 C（data→mmap）**：文件被 `mmap` 映射进地址空间后，**200MB 权重一个字节都不复制**，只要把每个 tensor 的 `data` 设成 `mmap_base + 它在 blob 里的 offset`，CPU 就能直接当内存读。这正是 03 章挂 110 个权重的方式。

> `no_alloc=true` 就表示"**别给 tensor 数据留内存**"——数据位置稍后用 mmap 指针填，context 只管 tensor 的"壳"。

### 2.4 绑定小结

```cpp
// view：一个 tensor 是另一个的视图时，data 指向源数据 + 偏移（共享同一块内存，无 memcpy）
void *data = view_src->data;
data = (char *)data + view_offs;
```

| 绑定 | 谁 → 指向哪儿 | 作用 |
|------|--------------|------|
| A | `ggml_object.offs` → `mem_buffer` 偏移 | 对象钉在池子内存 |
| B | `ggml_tensor` 直接放池子 | per-context 归属 |
| C | `tensor.data` → mmap_base+offset | 权重零拷贝 |
| D | `view.data` → `view_src->data + view_offs` | tensor 间共享/切片 |

## 三、tensor 形状：ne[] 与 nb[]（行主序）

```cpp
int64_t ne[GGML_MAX_DIMS]; // 每维元素数；ne[0] 是最内层（一行）
size_t  nb[GGML_MAX_DIMS]; // 每维字节步长（stride）
```

行主序下：`nb[0]=每元素字节数`，`nb[i]=nb[i-1]×ne[i-1]`。

**为什么同时存 ne 和 nb？** 只有 nb 能表达**非紧凑布局**（view 斜着切、量化 block）。邻接关系由 nb 决定，与 ne 无关。占字节数：

```
nbytes = (ne[0]/blck_size)×type_size × ne[1]×ne[2]×ne[3]
```

## 四、类型层：为何只 3 个、编号却照抄

```cpp
enum ggml_type {
    GGML_TYPE_F32  = 0,   // tinybrainbot 25 个 + Bonsai 353 个
    GGML_TYPE_F16  = 1,   // tinybrainbot 85 个
    GGML_TYPE_Q1_0 = 41,  // Bonsai 498 个（量化对照）
};
```

- **只 3 个**：面向的模型只用这三种，其余几十种量化用不到，不列。
- **编号照抄**：GGUF 里 tensor 的 `type` 是 `int32`，值 0/1/41 就是 F32/F16/Q1_0。**编号和文件不能错位**，是文件格式契约。

形状参数在 `ggml_type_traits`：`type_name`、`block_size`（每 block 元素数，未量化=1）、`type_size`（每 block 字节数）、`is_quantized`，加上**解量化/量化**函数指针 `to_float` / `from_float_ref`（属接口层，已声明，实现留给推理章）。主模型 F16 可直接位运算转 F32 参与计算（存 F16、算 F32）。

## 五、描述 cgraph（计算图，03 再用）

02 提前声明 `ggml_cgraph`，本节只描述不实现，让 03 建图时不用改头文件。

**为什么用"图"推理**：一次前向 = 一堆按依赖执行的 op（`mul_mat`、`add`、`rms_norm`…），必须**先算输入再算输出**。组织依赖的就是有向无环图（DAG）：

```
   x ───────────► rms_norm ─► mul_mat ─► add ─► …
   w(权重) ───────────────┘       ▲
                                  │     （w 是叶子，x 是叶子）
```

```cpp
struct ggml_cgraph {
    int size;    int n_nodes;  int n_leafs;
    struct ggml_tensor **nodes;   // 节点：每个 op 的输出，data 会变
    struct ggml_tensor **leafs;   // 叶子：常量/输入，data 不变
    int32_t *use_counts;          // 每个 tensor 被引用次数
    struct ggml_hash_set visited_hash_set; // 判重（防环）
    enum ggml_cgraph_eval_order order;     // 求值方向：左→右 / 右→左
    uint64_t uid;
};
```

- **nodes vs leafs**：叶子是起始输入（权重、prompt），节点是 op 产生的中间结果。求值 = 按依赖把节点一个个算出来。
- **visited_hash_set / use_counts**：一个 tensor 可能被多个 op 引用，求值时**每个节点只算一次**。靠 hash 判重 + 引用计数决定"输入齐了没、能不能算"。这是拓扑遍历的核心机制。
- **eval_order**：指定按什么方向遍历，把"模型结构"变成"可执行序列"。

> 03 会做：`ggml_build_forward` 塞节点 → 按 `order` 拓扑排序 → 逐个节点执行。02 只把蓝图的数据结构描述清楚。

## 六、代码结构与上游对照

```
02-ggml-context/
├── include/ggml.h  类型层：常量、enum ggml_type、ggml_type_traits、ggml_tensor、
│                   ggml_init_params、ggml_object_type、cgraph 相关声明（03 用）
├── src/ggml.cpp    私有定义 ggml_object/ggml_context + 池子分配（ggml_init/new_object…）
├── reference.md    参考源码对照
└── test/           手写测试（无第三方框架）
```

`ggml.h` 只声明公开结构体；`ggml.cpp` 藏着 `ggml_object`/`ggml_context` 的**内部定义**（内部字段只能在这改，别的 cpp 改了过不了编译）——**绑定只在拥有者手里**的 C 封装惯用法。

| 相对 `llama.cpp/` | 用途 |
|------|------|
| `ggml/include/ggml.h` | `ggml_tensor`/`ggml_init_params`/`ggml_type`/`ggml_cgraph` 声明 |
| `ggml/src/ggml.c` | `ggml_init`（建池）、`ggml_new_object`（池子分配）、`ggml_new_tensor_impl`（建张量+nb[]+view）、`ggml_set_name`、`ggml_nbytes`、`ggml_view_tensor` |
| `ggml/src/ggml-cpu/ggml-cpu.c` | F16/Q1_0 的 `to_float` 解量化（`ggml_table_f32_f16`） |
