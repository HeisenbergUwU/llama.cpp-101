// test-llama-io.cpp - 04 章「文件 IO 封装层」手写测试
// 验证 ①llama_file 打开并拿大小 ②llama_mmap 映射全文件 ③从 mmap 基址读出 GGUF magic（证明零拷贝指向文件内容）。

#include "llama-io.h"

#include <cstdio>
#include <cstring>
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

    // ---- 1. llama_file:打开 + 拿大小 ----
    llama::llama_file file(argv[1]);
    if (!file.valid)
    {
        return fail("llama_file 打开失败");
    }
    if (file.size == 0)
    {
        return fail("文件大小为 0,不合理");
    }
    std::printf("llama_file : fd=%d size=%zu\n", file.fd, file.size);

    // ---- 2. llama_mmap:映射整个文件 ----
    llama::llama_mmap mapping(file);
    if (mapping.addr == nullptr)
    {
        return fail("llama_mmap 映射失败");
    }
    if (mapping.size != file.size)
    {
        return fail("映射大小 != 文件大小");
    }
    std::printf("llama_mmap : addr=%p size=%zu\n", mapping.addr, mapping.size);

    // ---- 3. 校验零拷贝:从 mmap 基址读 4 字节应是 "GGUF" magic ----
    // GGUF 前 4 字节固定为 ASCII "GGUF"；能读出 = data 真指向文件内容，非悬空指针。
    char magic[5] = {0};
    std::memcpy(magic, mapping.addr, 4);
    if (std::strcmp(magic, "GGUF") != 0)
    {
        return fail("mmap 读到的前 4 字节不是 GGUF magic");
    }
    std::printf("magic     : %s (零拷贝读文件头 OK)\n", magic);

    std::printf("PASS 04 llama-io\n");
    return 0;
}
