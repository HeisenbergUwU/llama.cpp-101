// test-ggml-build-context.cpp - 03 章「迷你 ggml 加载层」的手写测试
//
// 约定（见 AGENTS.md）：不引入第三方测试框架；手写 main，退出码非 0 = 失败。
// 临时文件放 /tmp（本章无需文件）。
//
// 覆盖面（对应 src/ggml.cpp 的加载层 API）：
//   1. ggml_init / ggml_free：建池（自分配）+ 释放
//   2. ggml_new_tensor / _1d/_2d/_3d/_4d：各维度实例化
//   3. 池子切块：连续建 tensor，检查对象头 + 数据区互不重叠、对齐
//   4. nb[] 行主序步长换算
//   5. ggml_set_name：设名 & 截断
//   6. ggml_nbytes：字节数公式
//   7. no_alloc / view：数据区分配逻辑（三种 data 分支）
//   8. 池子不够：ggml_init / ggml_new_tensor 失败路径

#include "ggml.h"

#include <cstdio>
#include <cstdint>
#include <cstring>

// 失败宏：打印信息并以非零退出
#define CHECK(cond, ...)                                                    \
    do                                                                      \
    {                                                                       \
        if (!(cond))                                                        \
        {                                                                   \
            std::fprintf(stderr, "FAIL %s:%d  ", __FILE__, __LINE__);      \
            std::fprintf(stderr, __VA_ARGS__);                              \
            std::fprintf(stderr, "\n");                                     \
            return 1;                                                       \
        }                                                                   \
    } while (0)

// 当前第几个测试项，逐项打印进度，便于定位失败点
static int g_case = 0;
#define RUN_CASE(name)                                                      \
    do                                                                      \
    {                                                                       \
        ++g_case;                                                           \
        std::printf("[case %2d] %s\n", g_case, name);                       \
    } while (0)

int main()
{
    using namespace ggml;

    // ---- case 1：建池（内部自分配）+ 释放 ----
    RUN_CASE("ggml_init 自分配 + ggml_free");

    {
        ggml_init_params params = {};
        params.mem_size = 4096;
        params.no_alloc = false;
        ggml_context *ctx = ggml_init(params);
        CHECK(ctx != NULL, "ggml_init(4096) 应成功");
        ggml_free(ctx);
    }

    // ---- case 2：池子太小，ggml_init 失败 ----
    RUN_CASE("池子太小时 ggml_init 返回 NULL");

    {
        ggml_init_params params = {};
        params.mem_size = 1; // 小到放不下任何对象
        params.no_alloc = false;
        ggml_context *ctx = ggml_init(params);
        CHECK(ctx == NULL, "池子太小应建池失败");
    }

    // ---- case 3：各维度便捷封装 ----
    RUN_CASE("ggml_new_tensor_1d/2d/3d/4d 各维度实例化");

    {
        ggml_init_params params = {};
        params.mem_size = 1 << 16; // 64KB
        params.no_alloc = false;
        ggml_context *ctx = ggml_init(params);
        CHECK(ctx != NULL, "建池失败");

        ggml_tensor *t1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 10);
        ggml_tensor *t2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 8, 3);
        ggml_tensor *t3 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 4, 4);
        ggml_tensor *t4 = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 2, 3, 4, 5);
        CHECK(t1 && t2 && t3 && t4, "四种维度都应创建成功");

        // 维度数校验
        CHECK(t1->ne[0] == 10 && t1->ne[1] == 1 && t1->ne[2] == 1 && t1->ne[3] == 1,
              "1d 的 ne 应只填 ne0，其余为 1");
        CHECK(t4->ne[0] == 2 && t4->ne[1] == 3 && t4->ne[2] == 4 && t4->ne[3] == 5,
              "4d 的 ne 应全部填上");

        ggml_free(ctx);
    }

    // ---- case 4：nb[] 行主序步长换算 ----
    //   F32: nb[0]=4, nb[1]=4×ne0, nb[2]=nb[1]×ne1, nb[3]=nb[2]×ne2
    //   F16: nb[0]=2（type_size=sizeof(uint16_t)=2）
    RUN_CASE("nb[] 行主序步长换算");

    {
        ggml_init_params params = {};
        params.mem_size = 1 << 16;
        params.no_alloc = false;
        ggml_context *ctx = ggml_init(params);
        CHECK(ctx != NULL, "建池失败");

        // 2D F32, ne = [3, 5]
        ggml_tensor *t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 5);
        CHECK(t != NULL, "创建失败");
        CHECK(t->nb[0] == 4, "F32 nb[0] 应 = 4，实际 %zu", t->nb[0]);
        CHECK(t->nb[1] == 4 * 3, "nb[1] 应 = nb[0]×ne0 = 12，实际 %zu", t->nb[1]);
        CHECK(t->nb[2] == 12 * 5, "nb[2] 应 = nb[1]×ne1 = 60，实际 %zu", t->nb[2]);
        CHECK(t->nb[3] == 60 * 1, "nb[3] 应 = nb[2]×ne2 = 60，实际 %zu", t->nb[3]);

        // 1D F16, ne = [7]
        ggml_tensor *u = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 7);
        CHECK(u != NULL, "创建失败");
        CHECK(u->nb[0] == 2, "F16 nb[0] 应 = 2（sizeof uint16），实际 %zu", u->nb[0]);
        CHECK(u->nb[1] == 2 * 7, "F16 1d nb[1] 应 = 2×7 = 14，实际 %zu", u->nb[1]);

        ggml_free(ctx);
    }

    // ---- case 5：连续切块——对象互不重叠、链上对齐 ----
    RUN_CASE("连续建 tensor：池子切块不重叠且 16 字节对齐");

    {
        ggml_init_params params = {};
        params.mem_size = 1 << 16;
        params.no_alloc = false; // 数据也进池子
        ggml_context *ctx = ggml_init(params);
        CHECK(ctx != NULL, "建池失败");

        ggml_tensor *tensors[8];
        for (int i = 0; i < 8; i++)
        {
            tensors[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
            CHECK(tensors[i] != NULL, "第 %d 个 tensor 创建失败", i);
        }

        // 每个 tensor 结构指针与 data 都应按 16 对齐
        for (int i = 0; i < 8; i++)
        {
            CHECK(((uintptr_t)tensors[i] % 16) == 0,
                  "tensor[%d] 结构应 16 对齐，地址 %p", i, (void *)tensors[i]);
            CHECK(tensors[i]->data != NULL, "no_alloc=false 时 data 应已分配");
            CHECK(((uintptr_t)tensors[i]->data % 16) == 0,
                  "tensor[%d] data 应 16 对齐，地址 %p", i, tensors[i]->data);
        }

        // 互不重叠：后一个 tensor 的数据起点应 >= 前一个的 data 末尾
        for (int i = 1; i < 8; i++)
        {
            size_t prev_end = (size_t)tensors[i - 1]->data + ggml_nbytes(tensors[i - 1]);
            CHECK((size_t)tensors[i]->data >= prev_end ||
                      (size_t)tensors[i] >= prev_end,
                  "tensor[%d] 起点 %p 落在 tensor[%d] 数据区内", i,
                  (void *)tensors[i], i - 1);
        }

        ggml_free(ctx);
    }

    // ---- case 6：ggml_set_name 设名 + 截断 ----
    RUN_CASE("ggml_set_name 设名 & 超长截断");

    {
        ggml_init_params params = {};
        params.mem_size = 1 << 16;
        params.no_alloc = false;
        ggml_context *ctx = ggml_init(params);
        CHECK(ctx != NULL, "建池失败");

        ggml_tensor *t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
        CHECK(t != NULL, "创建失败");

        ggml_set_name(t, "token_embd.weight");
        CHECK(std::strcmp(t->name, "token_embd.weight") == 0,
              "设名应生效，实际 '%s'", t->name);

        // 超长名字应被截断且保证 '\0' 结尾
        char long_name[200];
        std::memset(long_name, 'A', sizeof(long_name));
        long_name[sizeof(long_name) - 1] = '\0';
        ggml_set_name(t, long_name);
        CHECK(std::strlen(t->name) == GGML_MAX_NAME - 1,
              "超长名字应截断到 %d 个字符，实际 %zu", GGML_MAX_NAME - 1, std::strlen(t->name));
        CHECK(t->name[GGML_MAX_NAME - 1] == '\0', "截断后必须以 '\\0' 结尾");

        ggml_free(ctx);
    }

    // ---- case 7：ggml_nbytes 字节数公式 ----
    //   未量化 => nbytes = ne0×type_size×ne1×ne2×ne3
    RUN_CASE("ggml_nbytes 字节数公式");

    {
        ggml_init_params params = {};
        params.mem_size = 1 << 16;
        params.no_alloc = false;
        ggml_context *ctx = ggml_init(params);
        CHECK(ctx != NULL, "建池失败");

        ggml_tensor *tf32 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 5);
        CHECK(tf32 != NULL && ggml_nbytes(tf32) == 3 * 5 * 4, "F32 2d nbytes = 60，实际 %zu",
              tf32 ? ggml_nbytes(tf32) : 0);

        ggml_tensor *tf16 = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 2, 3, 4);
        CHECK(tf16 != NULL && ggml_nbytes(tf16) == 2 * 3 * 4 * 2, "F16 3d nbytes = 48，实际 %zu",
              tf16 ? ggml_nbytes(tf16) : 0);

        ggml_free(ctx);
    }

    // ---- case 8：no_alloc=true 时 data 留空（04 章 mmap 会填）----
    RUN_CASE("no_alloc=true 时 data 应为 NULL");

    {
        ggml_init_params params = {};
        params.mem_size = 1 << 16;
        params.no_alloc = true;
        ggml_context *ctx = ggml_init(params);
        CHECK(ctx != NULL, "建池失败");

        ggml_tensor *t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
        CHECK(t != NULL, "创建失败");
        CHECK(t->data == NULL, "no_alloc=true 时 data 应为 NULL，实际 %p", t->data);

        ggml_free(ctx);
    }

    // ---- case 9：池子不够时 ggml_new_tensor 返回 NULL ----
    RUN_CASE("池子不足时 ggml_new_tensor 返回 NULL");

    {
        // 小池子：能建池，但不足以无限创建 tensor
        ggml_init_params params = {};
        params.mem_size = 512;
        params.no_alloc = false;
        ggml_context *ctx = ggml_init(params);
        CHECK(ctx != NULL, "建池失败");

        int ok = 0;
        while (ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 8) != NULL)
        {
            ok++;
            if (ok > 100)
            {
                break; // 防御：理论上 512B 很快耗尽
            }
        }
        CHECK(ok < 100, "512B 池子不应能创建 100 个 tensor");
        CHECK(ok >= 1, "512B 至少应能创建 1 个 tensor，实际 %d", ok);

        ggml_free(ctx);
    }

    // ---- case 10：view 分支语义（共享同一块池子内存）----
    // 公开 API 不暴露 view，这里验证 no_alloc=false 时池子里的 data 空间
    // 可读写（04 章 mmap 会在这个分支上做零拷贝挂载）。
    RUN_CASE("池子里 data 空间可读写（为 mmap 零拷贝铺路）");

    {
        ggml_init_params params = {};
        params.mem_size = 1 << 16;
        params.no_alloc = false;
        ggml_context *ctx = ggml_init(params);
        CHECK(ctx != NULL, "建池失败");

        ggml_tensor *src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
        CHECK(src != NULL && src->data != NULL, "源 tensor 创建失败");

        float *w = (float *)src->data;
        for (int i = 0; i < 16; i++)
        {
            w[i] = (float)i;
        }
        CHECK(w[0] == 0.0f && w[15] == 15.0f, "池子里 data 空间可读写");

        ggml_free(ctx);
    }

    std::printf("所有 %d 项用例通过\n", g_case);
    return 0;
}