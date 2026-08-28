// llama-io.cpp - 04 章「文件 IO 封装层」实现
//
// 只做一件事:把操作系统的 fopen/fseek/fread/fclose/mmap/munmap 藏起来，
// 对外暴露「文件」「映射」两个对象。上层不再出现任何裸系统调用。
//
// llama_file 对齐上游 llama.cpp/src/llama-mmap.cpp 的 llama_file::impl:
//   内部持 FILE*（用缓冲顺序读元数据），fd = fileno(fp)（供 mmap 零拷贝）。
// llama_mmap 对齐上游 llama_mmap::impl:
//   mmap(NULL, file->size(), PROT_READ, MAP_SHARED, fd, 0)
// 本教程去掉 prefetch/numa/MAP_POPULATE/madvise(那是性能优化,非原理)。

#include "llama-io.h"

#include <sys/mman.h> // mmap / munmap / PROT_READ / MAP_SHARED / MAP_FAILED
#include <sys/stat.h> // fstat / struct stat
#include <unistd.h>   // fileno / pread
#include <cstdio>     // fopen / fclose

namespace llama
{

    // ---- llama_file:fopen + fstat + 顺序读 + fclose ----
    llama_file::llama_file(const char *path)
        : size(0), valid(false)
    {
        // 1. fopen:带缓冲打开文件(读元数据用)，rb=只读二进制
        fp = fopen(path, "rb");
        if (fp == nullptr)
        {
            return; // valid 保持 false
        }

        // 2. fd = fileno(fp):从 FILE* 抠出底层文件描述符(mmap 要用整数 fd)
        fd = fileno(fp);

        // 3. fstat:拿文件大小(映射 / 越界校验都要知道文件多大)
        struct stat st;
        if (fstat(fd, &st) != 0)
        {
            fclose(fp);
            fp = nullptr;
            fd = -1;
            return; // valid 保持 false
        }

        size = (size_t)st.st_size;
        valid = true;
    }

    llama_file::~llama_file()
    {
        if (fp != nullptr)
        {
            fclose(fp);
        }
    }

    bool llama_file::read_at(uint64_t offset, void *out, size_t len) const
    {
        // pread 原子地读:不改变文件偏移，也不依赖 FILE* 缓冲，和多线程/mmap 混用安全
        if (offset + len > size)
        {
            return false; // 越界
        }
        return pread(fd, out, len, (off_t)offset) == (ssize_t)len;
    }

    // ---- llama_mmap:mmap ----
    llama_mmap::llama_mmap(const llama_file &file)
        : size(file.size)
    {
        // 对齐上游:零拷贝只读映射整个文件(MAP_SHARED 让多份映射共享文件页)
        addr = mmap(nullptr, file.size, PROT_READ, MAP_SHARED, file.fd, 0);
        if (addr == MAP_FAILED)
        {
            addr = nullptr;
        }
    }

    llama_mmap::~llama_mmap()
    {
        if (addr != nullptr)
        {
            munmap(addr, size);
        }
    }

} // namespace llama
