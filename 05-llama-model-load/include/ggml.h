// ggml.h - 迷你 ggml：数据结构声明 + 加载层函数接口（03 章）
//
// 这是教程 03 章的**接口层**：在 02 章类型层（ggml_type / ggml_tensor /
// ggml_context / ggml_object_type 等结构）之上，补充池子分配 API 的
// **函数声明**。实现放在 ggml.cpp。
//
// 命名与字段对齐上游 llama.cpp/ggml/include/ggml.h（结构体）与
// ggml/src/ggml.c（ggml_context / ggml_object 的内部定义）。
//
// 所有类型放进 `namespace ggml`，避免与 01 章的全局 `GGML_TYPE_*` 冲突。
//
// 阶段边界：
//   02 章 = 数据结构（ggml_type / ggml_tensor / ggml_context / 池子分配）
//   03 章 = 加载层：把池子分配真正写出来（ggml_init / ggml_new_tensor_* /
//           ggml_set_name / ggml_nbytes），让「往池子里实例化张量」可运行
//   04 章 = 用 03 的能力建 llama_model（GGUF -> 110 个 tensor + mmap 零拷贝）
//   推理  = 建图 / op（ggml_op / ggml_cgraph 等，后续章节再补）
//
// 面向的模型：tinybrainbot（讲解主模型，110 tensor）只用 F32 / F16，
// 所以 enum ggml_type 与 type_traits 只保留这两个类型。编号照抄上游
// （F32=0 / F16=1），和 GGUF 文件里的 int32 完全一致。

#pragma once

#include <stdint.h>
#include <stddef.h>

// ---- 常量（#define 无法进 namespace，故留在全局；都带 GGML_ 前缀不冲突）----
#define GGML_MAX_DIMS 4   // tensor 最多 4 维
#define GGML_MAX_NAME 64  // tensor 名字最长字节数（含 '\0'）
#define GGML_MEM_ALIGN 16 // 池子内存分配按 16 字节对齐（桌面 macOS/linux）

namespace ggml
{

    typedef void (*ggml_to_float_t)(void *x, float *y, int64_t k);
    typedef void (*ggml_from_float_t)(float *x, void *y, int64_t k);

    // ---- 张量数据类型 ----
    // 编号对应 GGUF 文件里 tensor info 的 type（int32），**必须与上游一致**。
    // 03 章面向 tinybrainbot（F16 主模型），只用 F32 / F16 两个：
    //   GGML_TYPE_F32  = 0（tinybrainbot 25 个）
    //   GGML_TYPE_F16  = 1（tinybrainbot 85 个）
    // 其余（Q1_0/Q4_K/Q6_K 等几十种量化）本章用不到，故不列。
    enum ggml_type
    {
        GGML_TYPE_F32 = 0,
        GGML_TYPE_F16 = 1,
    };

    // 每种类型的形状参数（blck_size / type_size 用来把「元素个数」换算成「字节数」）。
    // 字段取自上游 ggml_type_traits（ggml.h 2510 行）的**本项目用到的部分**：
    //   - blck_size / type_size / is_quantized / type_name
    // 上游还有一个 blck_size_interleave（量化数据重排优化，见 ggml-cpu/repack.cpp，
    // 与 tensor 布局 nb[] 无关），本项目不涉及，故不在此结构体中。
    struct ggml_type_traits
    {
        const char *type_name; // 人们可读的名字，如 "f32" / "f16"
        int64_t block_size;    // 每 block 的元素数（未量化类型为 1）
        size_t type_size;      // 每 block 的字节数（未量化类型 = 每元素字节数）
        bool is_quantized;     // 是否量化类型
        ggml_to_float_t to_float;
        ggml_from_float_t from_float_ref;
    };

    // ---- tensor ----
    // 一个 n 维张量的「结构 + 数据指针」。这是池子里的核心对象。
    //
    // 字段裁剪说明（对齐上游 ggml_tensor，ggml.h 680 行）：
    //   保留加载 / 布局所需：type、ne[]、nb[]、data、name，
    //   以及 mmap 零拷贝挂数据所需的 view_src / view_offs。
    //   暂不引入（后续推理 / 后端章节再补）：
    //     - op / op_params / src[] / flags（建图、op 语义）
    //     - buffer / extra（backend buffer 管理）
    struct ggml_tensor
    {
        enum ggml_type type; // 数据类型

        int64_t ne[GGML_MAX_DIMS]; // 每维元素数；ne[0] 是行内元素数（行主序）
        size_t nb[GGML_MAX_DIMS];  // 每维的字节步长（stride）：
                                   //   nb[0] = 每元素字节数
                                   //   nb[1] = nb[0] * (ne[0] / blck_size)
                                   //   nb[i] = nb[i-1] * ne[i-1]

        // 若这个 tensor 是某个 tensor 的 view（mmap 零拷贝挂数据用）：
        struct ggml_tensor *view_src; // 源 tensor（NULL 表示不是 view）
        size_t view_offs;             // 相对源 data 的字节偏移

        void *data; // 存放数据的指针（mmap 零拷贝时指向 mmap 基址 + 偏移）

        char name[GGML_MAX_NAME]; // 张量名，如 "token_embd.weight"
    };

    // ---- context ----
    // 只有声明（forward declaration）：ggml_context 是内部实现，字段藏在 ggml.cpp，
    // 外部只能拿到不透明句柄，通过函数接口操作。
    struct ggml_context;

    // ggml_init 的入参（对照上游 ggml.h 672 行）。
    struct ggml_init_params
    {
        size_t mem_size;  // 池子大小（字节）
        void *mem_buffer; // 若为 NULL，内部自行分配
        bool no_alloc;    // 是否不为 tensor 数据分配（本项目要 mmap 零拷贝 => true）
    };

    // ---- 池子里的对象类型 ----
    // 池子里每个对象（ggml_object，定义在 ggml.cpp 内部）都属于其中一种。
    enum ggml_object_type
    {
        GGML_OBJECT_TYPE_TENSOR,     // tensor 对象
        GGML_OBJECT_TYPE_GRAPH,      // 计算图对象
        GGML_OBJECT_TYPE_WORK_BUFFER // 工作缓冲区对象
    };

    // ==== 03 章·加载层函数接口 ====
    // 在 02 章类型层之上，把「往池子里实例化张量」的 API 声明出来。
    // 实现都在 ggml.cpp。对齐上游 ggml/src/ggml.c 的对应函数。

    // 建池：calloc 一块连续内存（params.mem_buffer==NULL 时内部自分配），
    // 初始化 objects_begin/end 空链表；返回不透明句柄，失败返回 NULL。
    ggml_context *ggml_init(struct ggml_init_params params);

    // 释放池子：若 mem_buffer 是内部自分配的（mem_buffer_owned），一并 free。
    void ggml_free(ggml_context *ctx);

    // 往池子里建一个 n_dims 维的 tensor（行主序，ne[] 是每维元素数）。
    // 成功返回池子内张量指针，失败返回 NULL。
    ggml_tensor *ggml_new_tensor(ggml_context *ctx, enum ggml_type type,
                                 int n_dims, const int64_t *ne);

    // 便捷封装：按维度数生成 ne[] 再丢给上面那个。
    ggml_tensor *ggml_new_tensor_1d(ggml_context *ctx, enum ggml_type type, int64_t ne0);
    ggml_tensor *ggml_new_tensor_2d(ggml_context *ctx, enum ggml_type type, int64_t ne0, int64_t ne1);
    ggml_tensor *ggml_new_tensor_3d(ggml_context *ctx, enum ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2);
    ggml_tensor *ggml_new_tensor_4d(ggml_context *ctx, enum ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);

    // 设张量名（拷进 tensor->name，超过 GGML_MAX_NAME-1 会截断）。返回 tensor。
    ggml_tensor *ggml_set_name(ggml_tensor *tensor, const char *name);

    // 按量化公式算 tensor 占用的字节数：
    //   (ne[0]/blck_size) × type_size × ne[1] × ne[2] × ne[3]
    size_t ggml_nbytes(const ggml_tensor *tensor);

    // ==== 以下为推理（建图遍历）阶段的结构，03 阶段已声明、暂不使用 ====
    // （保留，使后续建图章无需回头改头文件）

    // 推理时候图遍历方向
    enum ggml_cgraph_eval_order
    {
        GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT = 0,
        GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT,
        GGML_CGRAPH_EVAL_ORDER_COUNT
    };

    struct ggml_hash_set
    {
        size_t size;
        uint32_t *used;            // whether or not the keys are in use i.e. set
        struct ggml_tensor **keys; // actual tensors in the set, keys[i] is only defined if ggml_bitset_get(used, i)
    };

    struct ggml_cgraph
    {
        int size;    // maximum number of nodes/leafs/grads/grad_accs
        int n_nodes; // number of nodes currently in use
        int n_leafs; // number of leafs currently in use

        struct ggml_tensor **nodes; // tensors with data that can change if the graph is evaluated
        // eval only in this program, grad not needed

        struct ggml_tensor **leafs; // tensors with constant data
        int32_t *use_counts;        // number of uses of each tensor, indexed by hash table slot

        struct ggml_hash_set visited_hash_set;

        enum ggml_cgraph_eval_order order;

        uint64_t uid;
    };

} // namespace ggml
