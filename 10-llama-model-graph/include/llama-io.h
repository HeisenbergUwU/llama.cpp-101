// llama-io.h - 04 章「文件 IO 封装层」：把 open/fstat/mmap/close 等封装成 llama_file 与 llama_mmap 两类型。
// 对齐上游 llama-mmap.h 但去掉 pimpl 直接暴露成员。llama_file 持 FILE* 顺序读元数据 + 暴露 fd；llama_mmap 只映射整个文件。

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio> // FILE

namespace llama
{

    // ---- 文件封装:open + fstat + 顺序读 + close ----（对齐上游 llama_file，去 pimpl）
    //   ① 顺序缓冲读(read_at/read_le)——gguf 解析元数据用；② 裸 fd(供 mmap 零拷贝用)
    struct llama_file
    {
        FILE *fp = nullptr; // 带缓冲的文件句柄(顺序读元数据用;析构 fclose 它)
        int fd = -1;        // 底层文件描述符(fileno(fp)，供 mmap 零拷贝)
        size_t size;        // 文件字节数(fstat 的 st_size)
        bool valid;         // 打开是否成功(fopen/fstat 是否都 OK)

        // 打开文件(带缓冲)并 fstat 拿到大小。失败则 valid=false。
        // 不拷贝:文件句柄是独占资源。
        llama_file(const char *path);
        ~llama_file(); // fclose(fp)

        // 从绝对偏移 offset 读 len 字节到 out；恰好读完返回 true。
        bool read_at(uint64_t offset, void *out, size_t len) const;

        // 小端序读一个元素：从 offset 读 sizeof(T) 字节到 out。
        template <typename T>
        bool read_le(uint64_t offset, T &out) const
        {
            return read_at(offset, &out, sizeof(T));
        }

        llama_file(const llama_file &) = delete;
        llama_file &operator=(const llama_file &) = delete;
    };

    // ---- 内存映射封装:mmap + munmap ----（对齐上游 llama_mmap，去性能调优）
    // 只留核心：mmap 整个文件(PROT_READ|MAP_PRIVATE)、拿地址、析构时 munmap。
    struct llama_mmap
    {
        void *addr = nullptr; // mmap 返回的基址(页对齐)
        size_t size = 0;      // 映射字节数(= 文件大小)

        llama_mmap() = default; // 默认空映射(析构无操作)，供 llama_model 默认持有

        // 把整个文件按只读映射进地址空间。依赖 llama_file 已合法打开。
        explicit llama_mmap(const llama_file &file);
        ~llama_mmap(); // munmap(addr, size)

        // 移动:接管另一份映射的 addr/size,把源置空(避免双 munmap)。
        // 需要移动是因为 llama_model 会持有 llama_mmap(从局部临时 std::move 进来)。
        llama_mmap(llama_mmap &&other) noexcept;
        llama_mmap &operator=(llama_mmap &&other) noexcept;

        llama_mmap(const llama_mmap &) = delete;            // 用到了编译就报错
        llama_mmap &operator=(const llama_mmap &) = delete; // 用到了编译就报错
    };

} // namespace llama
