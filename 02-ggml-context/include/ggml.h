// ggml.h - 迷你 ggml：数据结构声明（02 章）
//
// 教程 02 章的类型层：只声明常量、枚举、结构体，不含任何函数接口或实现。
// 命名与字段对齐上游 llama.cpp/ggml/include/ggml.h 与 ggml/src/ggml.c。
// 所有类型放进 `namespace ggml`，避免与 01 章的全局 `GGML_TYPE_*` 冲突。
//
// 类型裁剪：主模型 tinybrainbot（110 tensor）只用 F32/F16，Bonsai（量化对照）
// 用 Q1_0+F32，故实际用到的类型只有三个。但 GGUF 里 tensor 的 type 是 int32，
// 枚举编号必须与文件一致，因此照抄上游，避免量化类型错位。

#pragma once

#include <stdint.h>
#include <stddef.h>

// ---- 常量（#define 无法进 namespace，故留在全局；都带 GGML_ 前缀不冲突）----
#define GGML_MAX_DIMS 4   // tensor 最多 4 维
#define GGML_MAX_NAME 64  // tensor 名字最长字节数（含 '\0'）
#define GGML_MEM_ALIGN 16 // 池子内存分配按 16 字节对齐（桌面 macOS/linux）

namespace ggml
{

    typedef void (*ggml_to_float_t)(const void *x, float *y, int64_t k);
    typedef void (*ggml_from_float_t)(const float *x, void *y, int64_t k);

    // ---- 张量数据类型 ----
    // 编号对应 GGUF 文件里 tensor info 的 type（int32），必须与上游一致。
    enum ggml_type
    {
        GGML_TYPE_F32 = 0,   // tinybrainbot 25 个 + Bonsai 353 个
        GGML_TYPE_F16 = 1,   // tinybrainbot 85 个
        GGML_TYPE_Q1_0 = 41, // Bonsai 498 个（量化对照）
    };

    // 每种类型的形状参数（blck_size / type_size 把「元素个数」换算成「字节数」）。
    // 字段取自上游 ggml_type_traits（ggml.h 2510 行）的本项目用到的部分。
    struct ggml_type_traits
    {
        const char *type_name; // 人们可读的名字，如 "f32" / "q1_0"
        int64_t block_size;    // 每 block 的元素数（未量化类型为 1）
        size_t type_size;      // 每 block 的字节数（未量化类型 = 每元素字节数）
        bool is_quantized;     // 是否量化类型
        ggml_to_float_t to_float;
        ggml_from_float_t from_float_ref;
    };

    // ---- tensor ----
    // 一个 n 维张量的「结构 + 数据指针」，是池子里的核心对象。
    // 字段对齐上游 ggml_tensor（ggml.h 680 行），保留加载/布局所需，
    // 暂不引入 op/src[]/buffer 等建图、后端字段（后续章节再补）。
    struct ggml_tensor
    {
        enum ggml_type type; // 数据类型

        int64_t ne[GGML_MAX_DIMS]; // 每维元素数；ne[0] 是行内元素数（行主序）
        size_t nb[GGML_MAX_DIMS];  // 每维字节步长：nb[0]=每元素字节数，nb[i]=nb[i-1]*ne[i-1]

        // mmap 零拷贝挂数据用：若此 tensor 是另一个的 view
        struct ggml_tensor *view_src; // 源 tensor（NULL 表示不是 view）
        size_t view_offs;             // 相对源 data 的字节偏移

        void *data; // 数据指针（mmap 零拷贝时指向 mmap 基址 + 偏移）

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

} // namespace ggml
