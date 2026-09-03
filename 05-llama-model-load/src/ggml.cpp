#include "ggml.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// ggml_object / ggml_context 是 ggml 内部实现：字段藏在这里，
// 外部只拿不透明句柄，看不到内部字段（别的 cpp 直接改会编译不过）。

namespace ggml
{

    // 池子分块台账：每切出一块内存就登记一个 ggml_object。
    // 内存布局：[ ggml_object 头 ][ 载荷 payload ]
    struct ggml_object
    {
        size_t offs; // 载荷起点相对 mem_buffer 的偏移（跳过对象头之后的载荷区）
        size_t size; // 载荷大小（已 ggml_align_up 到 GGML_MEM_ALIGN=16 倍数）

        struct ggml_object *next; // 链表下一块对象头；经 ctx->objects_begin/end 串成单链表

        enum ggml_object_type type; // 这块是什么：TENSOR / GRAPH / WORK_BUFFER

        char padding[4]; // 把对象头大小补到 16 倍数，使「载荷起点链条」保持 16 字节对齐
    };

    static const size_t GGML_OBJECT_SIZE = sizeof(struct ggml_object);

    struct ggml_context
    {
        size_t mem_size;
        void *mem_buffer;
        bool mem_buffer_owned;
        bool no_alloc;

        int n_objects;

        struct ggml_object *objects_begin;
        struct ggml_object *objects_end;
    };

    // ---- 类型参数表：F32 / F16（03 章只加载、不做量化）----
    // to_float / from_float_ref 的解量化实现留到推理章，这里先空着。
    static const ggml_type_traits type_traits[] = {
        /* 0 = F32 */ {"f32", 1, sizeof(float), false, NULL, NULL},
        /* 1 = F16 */ {"f16", 1, sizeof(uint16_t), false, NULL, NULL},
    };

    // 向上取整到 GGML_MEM_ALIGN 的倍数（保证对象按 16 字节对齐存放）。
    inline static size_t ggml_align_up(size_t n)
    {
        const size_t remainder = n % GGML_MEM_ALIGN;
        if (remainder == 0)
        {
            return n;
        }
        return n + (GGML_MEM_ALIGN - remainder);
    }

    // ---- 建池 ----
    // 对齐上游 ggml_init：句柄 ggml_context 单独 malloc 不占池子（只是管理器），mem_buffer 只放对象/tensor；mem_buffer 为 NULL 则内部 calloc。
    ggml_context *ggml_init(struct ggml_init_params params)
    {
        // 句柄单独分配（这是「不透明句柄」的背面：调用方只见指针，
        // 见不到这里其实是一块独立 malloc 的结构体）。
        ggml_context *ctx = (ggml_context *)malloc(sizeof(ggml_context));
        if (ctx == NULL)
        {
            return NULL;
        }

        // 池子连一个对象头都放不下，建不起来。
        if (params.mem_size < GGML_OBJECT_SIZE)
        {
            free(ctx);
            return NULL;
        }

        ctx->mem_buffer = params.mem_buffer;
        if (ctx->mem_buffer == NULL)
        {
            // 未显式给 buffer：内部 calloc 清零自分配，并标记归自己管。
            ctx->mem_buffer = calloc(1, params.mem_size);
            if (ctx->mem_buffer == NULL)
            {
                free(ctx);
                return NULL;
            }
            ctx->mem_buffer_owned = true;
        }
        else
        {
            ctx->mem_buffer_owned = false;
        }

        ctx->mem_size = params.mem_size;
        ctx->no_alloc = params.no_alloc;
        ctx->n_objects = 0;
        ctx->objects_begin = NULL;
        ctx->objects_end = NULL;

        return ctx;
    }
    // ---- 释放池子 ----
    void ggml_free(ggml_context *ctx)
    {
        if (ctx == NULL)
        {
            return;
        }
        if (ctx->mem_buffer_owned)
        {
            free(ctx->mem_buffer);
        }
        // 句柄是单独 malloc 的，一并释放。
        free(ctx);
    }

    // ---- 池子切块：在末尾追加一块，返回对象头 ----
    // 布局（对齐上游 ggml_new_object）：[ ggml_object 头 ][ 载荷 size ]，obj->offs 指载荷起点，obj->size 是对齐后载荷大小，下一个对象接在载荷之后。
    static ggml_object *ggml_new_object(ggml_context *ctx, enum ggml_object_type type, size_t size)
    {
        // 载荷向上取整到 16
        const size_t size_needed = ggml_align_up(size);

        // 上一个载荷的末尾 = 本对象头的位置；第一个对象从池子起点开始
        size_t cur_end = ctx->objects_end
                             ? ctx->objects_end->offs + ctx->objects_end->size
                             : 0;

        if (cur_end + GGML_OBJECT_SIZE + size_needed > ctx->mem_size)
        {
            return NULL; // 池子不够
        }

        ggml_object *obj = (ggml_object *)((char *)ctx->mem_buffer + cur_end);
        obj->offs = cur_end + GGML_OBJECT_SIZE; // 载荷起点
        obj->size = size_needed;
        obj->next = NULL;
        obj->type = type;

        if (ctx->objects_end != NULL)
        {
            ctx->objects_end->next = obj;
        }
        else
        {
            ctx->objects_begin = obj;
        }
        ctx->objects_end = obj;
        ctx->n_objects++;

        return obj;
    }

    // ---- 核心：实例化一个张量（对齐上游 ggml_new_tensor_impl）----
    // 载荷 = [ ggml_tensor 结构 ] + [ 数据 bytes（no_alloc=false 且非 view 时）]
    static ggml_tensor *ggml_new_tensor_impl(ggml_context *ctx, enum ggml_type type,
                                             int n_dims, const int64_t *ne,
                                             ggml_tensor *view_src, size_t view_offs)
    {
        if (n_dims < 1 || n_dims > GGML_MAX_DIMS)
        {
            return NULL;
        }
        const ggml_type_traits &tt = type_traits[(int)type];

        // 数据字节数 = (ne[0]/blk_size)×type_size × ne[1]×…×ne[n_dims-1]
        // 对齐上游：第一行已含 ne[0]，故后续维度从 i=1 开始乘，不再重复 ne[0]。
        size_t data_size = (size_t)(ne[0] / tt.block_size) * tt.type_size; // row
        for (int i = 1; i < n_dims; i++)                                   // rest dims
        {
            data_size *= (size_t)ne[i];
        }

        // 只有「非 view 且 !no_alloc」才把数据也塞进池子，否则数据另寻（mmap）
        size_t obj_alloc_size = (view_src == NULL && !ctx->no_alloc) ? data_size : 0;

        ggml_object *obj = ggml_new_object(ctx, GGML_OBJECT_TYPE_TENSOR,
                                           sizeof(ggml_tensor) + obj_alloc_size);
        if (obj == NULL)
        {
            return NULL;
        }

        ggml_tensor *t = (ggml_tensor *)((char *)ctx->mem_buffer + obj->offs);

        t->type = type;
        for (int i = 0; i < GGML_MAX_DIMS; i++)
        {
            t->ne[i] = 1;
        }
        for (int i = 0; i < n_dims; i++)
        {
            t->ne[i] = ne[i];
        }

        // 行主序步长：nb[0]=每元素字节数；nb[1]=nb[0]×(ne[0]/blk)；nb[i]=nb[i-1]×ne[i-1]
        t->nb[0] = tt.type_size;
        t->nb[1] = t->nb[0] * (t->ne[0] / tt.block_size);
        for (int i = 2; i < GGML_MAX_DIMS; i++)
        {
            t->nb[i] = t->nb[i - 1] * t->ne[i - 1];
        }

        t->view_src = view_src;
        t->view_offs = view_offs;
        if (obj_alloc_size > 0)
        {
            // 数据紧跟 tensor 结构（绑定「池子内分配」）
            t->data = (void *)(t + 1);
        }
        else if (view_src != NULL)
        {
            // view：data 指向源数据 + 偏移（绑定 D，共享内存）
            t->data = (char *)view_src->data + view_offs;
        }
        else
        {
            // no_alloc=true：data 留空，04 章用 mmap 指针填
            t->data = NULL;
        }

        return t;
    }

    // ---- 公开入口 ----
    ggml_tensor *ggml_new_tensor(ggml_context *ctx, enum ggml_type type,
                                 int n_dims, const int64_t *ne)
    {
        return ggml_new_tensor_impl(ctx, type, n_dims, ne, NULL, 0);
    }

    ggml_tensor *ggml_new_tensor_1d(ggml_context *ctx, enum ggml_type type, int64_t ne0)
    {
        const int64_t ne[] = {ne0};
        return ggml_new_tensor(ctx, type, 1, ne);
    }

    ggml_tensor *ggml_new_tensor_2d(ggml_context *ctx, enum ggml_type type, int64_t ne0, int64_t ne1)
    {
        const int64_t ne[] = {ne0, ne1};
        return ggml_new_tensor(ctx, type, 2, ne);
    }

    ggml_tensor *ggml_new_tensor_3d(ggml_context *ctx, enum ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2)
    {
        const int64_t ne[] = {ne0, ne1, ne2};
        return ggml_new_tensor(ctx, type, 3, ne);
    }

    ggml_tensor *ggml_new_tensor_4d(ggml_context *ctx, enum ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3)
    {
        const int64_t ne[] = {ne0, ne1, ne2, ne3};
        return ggml_new_tensor(ctx, type, 4, ne);
    }

    // ---- 设名字 ----
    ggml_tensor *ggml_set_name(ggml_tensor *tensor, const char *name)
    {
        snprintf(tensor->name, GGML_MAX_NAME, "%s", name);
        return tensor;
    }

    // ---- 算字节数：对照 01 章 nbytes 公式 ----
    size_t ggml_nbytes(const ggml_tensor *tensor)
    {
        const ggml_type_traits &tt = type_traits[(int)tensor->type];
        if (tensor->ne[0] == 0)
        {
            return 0;
        }
        size_t nbytes = (size_t)(tensor->ne[0] / tt.block_size) * tt.type_size;
        for (int i = 1; i < GGML_MAX_DIMS; i++)
        {
            nbytes *= (size_t)tensor->ne[i];
        }
        return nbytes;
    }

} // namespace ggml
