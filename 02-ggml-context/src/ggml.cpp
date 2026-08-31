#include "ggml.h"
#include <cstdint>
#include <cstdio>

// ggml_object / ggml_context 是 ggml 内部实现：字段藏在这里，
// 外部只拿不透明句柄，看不到内部字段（别的 cpp 直接改会编译不过）。

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
