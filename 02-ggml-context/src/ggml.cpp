#include "ggml.h"
#include <cstdint>
#include <cstdio>

// 只能在 ggml 中进行修改结构体内部变量，其他 cpp 文件如果修改一下结构体变量将不能通过编译。
// ggml_object / ggml_context 是 ggml 内部实现(藏在 namespace ggml 里)，
// 外部只拿到 ggml_context 的不透明句柄，看不到内部字段。

namespace ggml
{

    struct ggml_object
    {
        size_t offs;
        size_t size;

        struct ggml_object *next;

        enum ggml_object_type type;

        char padding[4];
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

} // namespace ggml
