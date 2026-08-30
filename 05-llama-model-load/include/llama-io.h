// llama-io.h - 04 章「文件 IO 封装层」接口
//
// 把裸操作系统调用(open/fstat/mmap/munmap/close/fread/fseek)封装成两个小类型，
// 让上层(gguf / llama_model)不直接碰系统调用，只看得到「文件」「映射」两个概念。
// 对齐上游 llama.cpp/src/llama-mmap.h 里的 llama_file 与 llama_mmap，
// 但教程版去掉了 pimpl，直接暴露成员，方便读者看穿每一层。
//
// llama_file 与上游一致：内部持有 FILE*（用缓冲顺序读元数据），
// 同时暴露底层 fd（用 fileno 取出，供 mmap 零拷贝用）——一份句柄两种用法。
// llama_mmap 只负责把整个文件映射进地址空间。
//
// 为什么单独抽这一层(呼应本项目分层):
//   - 逻辑计算(ggml 建 tensor、gguf 解析)不该掺和 OS 细节。
//   - mmap 是"把文件零拷贝挂进地址空间"的唯一手段，值得单独讲清。
//   01(gguf)/03(ggml) 被拉进 04 章后，文件操作都改走这一层。

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio> // FILE

namespace llama
{

    // ---- 文件封装:open + fstat + 顺序读 + close ----
    // 对齐上游 struct llama_file(llama-mmap.h:16)。教程去掉 pimpl，
    // 但保留两种能力:
    //   ① 顺序缓冲读(read_at / read_le)——gguf 解析元数据用
    //   ② 裸 fd(供 mmap)——mmap 零拷贝用
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
    };

    // ---- 内存映射封装:mmap + munmap ----
    // 对齐上游 struct llama_mmap(llama-mmap.h:43)。
    // 教程简化:去掉 prefetch/numa/MAP_POPULATE/madvise 等性能调优，
    // 只留核心:mmap 整个文件(PROT_READ | MAP_PRIVATE)、拿地址、析构时 munmap。
    struct llama_mmap
    {
        void *addr = nullptr; // mmap 返回的基址(页对齐)
        size_t size = 0;      // 映射字节数(= 文件大小)

        llama_mmap() = default; // 默认空映射(析构无操作)，供 llama_model 默认持有

        // 把整个文件按只读映射进地址空间。依赖 llama_file 已合法打开。
        explicit llama_mmap(const llama_file &file);
        ~llama_mmap(); // munmap(addr, size)
    };

} // namespace llama
