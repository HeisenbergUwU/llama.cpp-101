// test-load-gguf.cpp - 01 章演示/测试入口
//
// 用法：
//   g++ -std=c++11 -Iinclude src/gguf-loader.cpp tests/test-load-gguf.cpp -o test-load-gguf
//   ./test-load-gguf ../resources/tinybrainbot-100m-v3-instruct/tinybrainbot-100m-v3-instruct-f16.gguf
//
// 功能：加载并校验一个 GGUF 文件，打印 header、全部 KV 与前几个 tensor，
//       全部通过打印 "OK: N tensors, M kv pairs checked" 并返回 0，
//       失败打印错误并返回非 0（与 llama.cpp 测试约定一致）。

#include "gguf-loader.h"

#include <cstdio>
#include <cstring>
#include <string>

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }

    gguf::file_info info;
    std::string err;
    if (!gguf::load_and_check(argv[1], info, err))
    {
        fprintf(stderr, "FAIL: %s\n", err.c_str());
        return 1;
    }

    printf("GGUF file: %s\n", argv[1]);
    printf("  version    : %u\n", info.version);
    printf("  n_tensors  : %lld\n", (long long)info.n_tensors);
    printf("  n_kv       : %lld\n", (long long)info.n_kv);
    printf("  alignment  : %u\n", info.alignment);
    printf("  data_offset: %llu (0x%llx)\n",
           (unsigned long long)info.data_offset, (unsigned long long)info.data_offset);
    printf("  file_size  : %llu bytes\n", (unsigned long long)info.file_size);

    printf("\nAll KV pairs:\n");
    for (size_t i = 0; i < info.kv.size(); ++i)
    {
        const auto &kv = info.kv[i];
        printf("  [%zu] %-32s = %s\n", i, kv.key.c_str(), gguf::fmt_value(kv).c_str());
    }

    printf("\nFirst tensors:\n");
    size_t shown = info.tensors.size() < 100 ? info.tensors.size() : 100;
    for (size_t i = 0; i < shown; ++i)
    {
        const auto &t = info.tensors[i];
        printf("  [%zu] %-24s type=%lld ne=(%lld, %lld, %lld, %lld) off=%llu nbytes=%lld\n",
               i, t.name.c_str(), (long long)t.type,
               (long long)t.ne[0], (long long)t.ne[1], (long long)t.ne[2], (long long)t.ne[3],
               (unsigned long long)t.offset, (long long)t.nbytes);
    }

    printf("\nOK: %lld tensors, %lld kv pairs checked\n",
           (long long)info.tensors.size(), (long long)info.kv.size());
    return 0;
}
