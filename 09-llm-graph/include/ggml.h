// ggml.h - 迷你 ggml：数据结构声明 + 加载层函数接口（03 章）
// 在 02 章类型层之上补池子分配 API 声明，实现放 ggml.cpp；命名字段对齐上游，进 namespace ggml 避免与 01 章全局 GGML_TYPE_* 冲突；只面向 F32/F16，编号照抄上游（F32=0/F16=1）。

#pragma once

#include <stdint.h>
#include <stddef.h>

// ---- 常量（#define 无法进 namespace，故留在全局；都带 GGML_ 前缀不冲突）----
#define GGML_MAX_DIMS 4       // tensor 最多 4 维
#define GGML_MAX_NAME 64      // tensor 名字最长字节数（含 '\0'）
#define GGML_MEM_ALIGN 16     // 池子内存分配按 16 字节对齐（桌面 macOS/linux）
#define GGML_MAX_SRC 8        // 一个算子节点的最多输入数
#define GGML_MAX_OP_PARAMS 64 // op 专属参数缓冲字节数（= int32_t[16]）

namespace ggml
{

    typedef void (*ggml_to_float_t)(void *x, float *y, int64_t k);
    typedef void (*ggml_from_float_t)(float *x, void *y, int64_t k);

    // ---- 张量数据类型 ----
    // 编号对应 GGUF 文件里 tensor info 的 type（int32），必须与上游一致。
    enum ggml_type
    {
        GGML_TYPE_F32 = 0, // tinybrainbot 25 个
        GGML_TYPE_F16 = 1, // tinybrainbot 85 个
    };

    // 每种类型的形状参数（blck_size / type_size 把「元素个数」换算成「字节数」）。
    // 字段取自上游 ggml_type_traits（ggml.h 2510 行）的本项目用到的部分。
    struct ggml_type_traits
    {
        const char *type_name; // 人们可读的名字，如 "f32" / "f16"
        int64_t block_size;    // 每 block 的元素数（未量化类型为 1）
        size_t type_size;      // 每 block 的字节数（未量化类型 = 每元素字节数）
        bool is_quantized;     // 是否量化类型
        ggml_to_float_t to_float;
        ggml_from_float_t from_float_ref;
    };

    // ---- 算子类型 ----
    // NONE=叶子（权重/输入）；其余为算子节点，对齐上游 GGML_OP_*，只保留通用算子（模型专属的 embed/attention 不进图设施）。
    enum ggml_op
    {
        GGML_OP_NONE = 0,
        GGML_OP_RMS_NORM,
        GGML_OP_MUL_MAT,
        GGML_OP_ROPE,
        GGML_OP_SILU,
        GGML_OP_MUL,
        GGML_OP_ADD,
    };

    // ---- tensor ----
    // 一个 n 维张量的「结构 + 数据指针」，池子里的核心对象，字段对齐上游 ggml_tensor（ggml.h 680 行）；既是权重/输入叶子（op=NONE）也是算子节点（op 非 NONE，src[] 记输入、data 存结果），和上游一致。
    struct ggml_tensor
    {
        enum ggml_type type; // 数据类型

        int64_t ne[GGML_MAX_DIMS]; // 每维元素数；ne[0] 是行内元素数（行主序）
        size_t nb[GGML_MAX_DIMS];  // 每维字节步长：nb[0]=每元素字节数，nb[i]=nb[i-1]*ne[i-1]

        // mmap 零拷贝挂数据用：若此 tensor 是另一个的 view
        struct ggml_tensor *view_src; // 源 tensor（NULL 表示不是 view）
        size_t view_offs;             // 相对源 data 的字节偏移

        // 算子信息：op=算子类型；op_params=op 专属标量参数（按 op 约定槽位，
        // 写用 ggml_set/get_op_params_i32/f32）；src[]=输入。
        enum ggml_op op;
        int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
        struct ggml_tensor *src[GGML_MAX_SRC];

        void *data; // 数据指针（mmap 零拷贝时指向 mmap 基址 + 偏移）

        char name[GGML_MAX_NAME]; // 张量名，如 "token_embd.weight"
    };

    // ---- op_params 读写 helper（对齐上游 ggml_set/get_op_params_i32/f32）----
    void ggml_set_op_params_i32(struct ggml_tensor *a, int slot, int32_t v);
    int32_t ggml_get_op_params_i32(const struct ggml_tensor *a, int slot);
    void ggml_set_op_params_f32(struct ggml_tensor *a, int slot, float v);
    float ggml_get_op_params_f32(const struct ggml_tensor *a, int slot);

    // ---- context ----
    // 只有 forward declaration：ggml_context 是内部实现，字段藏在 ggml.cpp，外部只能拿不透明句柄，通过函数接口操作。
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
    // 在 02 章类型层之上，声明「往池子里实例化张量」的 API，实现都在 ggml.cpp，对齐上游 ggml/src/ggml.c。

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

} // namespace ggml
