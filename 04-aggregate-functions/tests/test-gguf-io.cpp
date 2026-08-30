// test-gguf-io.cpp - 04 章「gguf 适配走 llama-io」手写测试
//
// 约定(见 AGENTS.md):手写 main,退出码非 0 = 失败。
// 目标:验证把 01 章 gguf.cpp 拉进 04 章、并改造其文件操作改用 llama_file 后，
//   解析结果与原来完全一致(不再裸调 fopen/fread,全部走 04 的 IO 封装层)。
//   用 tinybrainbot 实测:magic / version / n_tensors / n_kv / data_offset 都对。

#include "gguf.h"      // 01 章解析器(已改走 llama_file)
#include "llama-io.h"  // 04 章 IO 封装层(gguf 现在依赖它)

#include <cstdio>
#include <string>

namespace
{
    int fail(const char *msg)
    {
        std::fprintf(stderr, "FAIL %s\n", msg);
        return 1;
    }
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "用法: %s <model.gguf>\n", argv[0]);
        return 2;
    }

    gguf::gguf_context info;
    std::string err;

    // gguf_load 复用上层打开的 llama_file 解析（不再自己 fopen）
    llama::llama_file file(argv[1]);
    if (!gguf::gguf_load(file, info, err))
    {
        return fail(err.c_str());
    }

    // 已知事实(AGENTS.md 实测记录,tinybrainbot):
    //   magic=GGUF, version=3, n_tensors=110, n_kv=32, data_offset=0xbaf80
    if (info.n_tensors != 110)
    {
        return fail("n_tensors 应为 110");
    }
    if (info.n_kv != 32)
    {
        return fail("n_kv 应为 32");
    }
    if (info.offset != 0xbaf80)
    {
        return fail("data_offset 应为 0xbaf80");
    }

    std::printf("gguf_load(走 llama_file) 解析成功:\n");
    std::printf("  n_tensors = %lld   n_kv = %lld\n",
                (long long)info.n_tensors, (long long)info.n_kv);
    std::printf("  data_offset = 0x%llx   file_size = %llu\n",
                (unsigned long long)info.offset,
                (unsigned long long)info.file_size);
    std::printf("  首 tensor 名 = %s\n", info.info[0].name.c_str());

    std::printf("PASS 04 gguf-io\n");
    return 0;
}
