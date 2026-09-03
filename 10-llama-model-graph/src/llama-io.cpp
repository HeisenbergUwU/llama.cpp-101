// llama-io.cpp - 04 章「文件 IO 封装层」：把 fopen/fread/mmap 等裸系统调用藏起来，只暴露「文件」「映射」。
// llama_file 对齐 llama_file::impl（FILE* 读元数据、fd 供 mmap）；llama_mmap 对齐 impl(mmap)；去掉性能调优。

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

    // ---- llama_mmap:移动构造/移动赋值 ---- 把已 mmap 的映射移交给另一 llama_mmap（如移入 llama_model.mmap）。
    // 接管后必须把源置空，否则两者析构会对同一地址 munmap 两次。
    llama_mmap::llama_mmap(llama_mmap &&other) noexcept
        : addr(other.addr), size(other.size)
    {
        other.addr = nullptr;
        other.size = 0;
    }

    llama_mmap &llama_mmap::operator=(llama_mmap &&other) noexcept
    {
        if (this != &other)
        {
            // 先释放自己手上已有的映射，再接管对方的
            if (addr != nullptr)
            {
                munmap(addr, size);
            }
            addr = other.addr;
            size = other.size;
            other.addr = nullptr;
            other.size = 0;
        }
        return *this;
    }

} // namespace llama
