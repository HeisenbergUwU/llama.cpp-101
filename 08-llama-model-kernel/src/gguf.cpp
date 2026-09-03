// gguf.cpp - GGUF 文件裸解析器实现（load + check）。读取对照上游 gguf_init_from_file，
// 校验对照 llama_tensor_weight；省去端序/多分片/mmap，只讲「怎么读 + 怎么查」。

#include "gguf.h"
#include "llama-io.h" // 04/05 章：文件 IO 封装层（用 llama_file 读，不再裸调 fopen/fread）

#include <cstring>
#include <sstream>

namespace gguf
{

    // 游标式顺序读取辅助（与上游 gguf.cpp static helper 一致）：调 llama_file::read_at，
    // 这里只补「游标推进」，小端序读一个元素。
    template <typename T>
    static bool read_le(const llama::llama_file &file, uint64_t &pos, T &out)
    {
        if (!file.read_at(pos, &out, sizeof(T)))
            return false;
        pos += sizeof(T);
        return true;
    }

    // 读一个 GGUF 字符串：[u64 长度][len 个字节]（不含结尾 '\0'），读完推进游标 pos。
    static bool read_string(const llama::llama_file &file, uint64_t &pos, std::string &out, std::string &err, const char *what, uint64_t file_size)
    {
        uint64_t len = 0;
        if (!read_le(file, pos, len))
        {
            err = std::string("unexpected EOF reading ") + what + " length";
            return false;
        }
        // 越界校验：len 不能超过文件大小，读到末尾也不能越过文件尾（防损坏文件）
        if (len > file_size || pos + len > file_size)
        {
            err = std::string(what) + " length out of bounds";
            return false;
        }
        out.resize(len);
        // &out[0] 是 std::string 内部缓冲区的起始地址
        if (!file.read_at(pos, &out[0], len))
        {
            err = std::string("unexpected EOF reading ") + what;
            return false;
        }
        pos += len;
        return true;
    }

    static uint64_t pad(uint64_t x, uint32_t a) { return (x + a - 1) / a * a; }

    // 返回各 gguf_type 的标量字节长度；STRING/ARRAY 不是标量返回 0，未知类型返回 -1
    static int scalar_size(gguf_type t)
    {
        switch (t)
        {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            return 1;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            return 2;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            return 4;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            return 8;
        case GGUF_TYPE_STRING:
        case GGUF_TYPE_ARRAY:
            return 0;
        case GGUF_TYPE_COUNT:
            break;
        }
        return -1;
    }

    // 打印 KV 值（标量/字符串/数组），转成可读字符串。
    // 数组最多展开前 20 个元素，多余用 "..." 省略并标注总数，避免大数组（如几万字符串的词表）刷屏。
    std::string fmt_value(const gguf_kv &kv)
    {
        if (!kv.is_array())
            return kv.value;
        const size_t n = kv.arr_value.size();
        const size_t head = n < 20 ? n : 20;
        std::ostringstream os;
        os << "[";
        for (size_t i = 0; i < head; ++i)
        {
            if (i)
                os << ", ";
            os << kv.arr_value[i];
        }
        if (n > head)
            os << ", ...(共 " << n << " 个)";
        os << "]";
        return os.str();
    }

    // 读取 KV 段的泛型元素（标量或数组）。这里为了通用，把每个元素读成字符串。
    static bool read_kv_pairs(const llama::llama_file &file, uint64_t &pos, int64_t n_kv, gguf_context &info,
                              std::string &err, uint64_t file_size)
    {
        for (int64_t i = 0; i < n_kv; ++i)
        {
            gguf_kv kv;
            if (!read_string(file, pos, kv.key, err, "kv key", file_size))
                return false;

            // 重复 key 检查（上游 gguf.cpp 有同样检查）
            for (const auto &e : info.kv)
            {
                if (e.key == kv.key)
                {
                    err = "duplicate kv key: '" + kv.key + "'";
                    return false;
                }
            }

            int32_t vt = 0;
            if (!read_le(file, pos, vt))
            {
                err = "unexpected EOF reading kv value type";
                return false;
            }
            kv.type = vt;

            if (vt == GGUF_TYPE_ARRAY)
            {
                int32_t et = 0;
                uint64_t n = 0;
                if (!read_le(file, pos, et))
                {
                    err = "unexpected EOF reading array type";
                    return false;
                }
                if (!read_le(file, pos, n))
                {
                    err = "unexpected EOF reading array count";
                    return false;
                }
                if ((uint64_t)n > file_size)
                {
                    err = "array count out of bounds";
                    return false;
                }
                kv.arr_type = et;
                for (uint64_t j = 0; j < n; ++j)
                {
                    if (et == GGUF_TYPE_STRING)
                    {
                        std::string tmp;
                        if (!read_string(file, pos, tmp, err, "array string", file_size))
                            return false;
                        kv.arr_value.push_back(std::move(tmp));
                    }
                    else
                    {
                        uint8_t buf[8] = {0};
                        int sz = scalar_size((gguf_type)et);
                        if (sz <= 0)
                        {
                            err = "unsupported array element type " + std::to_string(et);
                            return false;
                        }
                        if (pos + (uint64_t)sz > file_size)
                        {
                            err = "array data out of bounds";
                            return false;
                        }
                        if (!file.read_at(pos, buf, sz))
                        {
                            err = "unexpected EOF reading array element";
                            return false;
                        }
                        pos += sz;
                        // 二进制值转可读字符串（f32/f64 直接读成数值）
                        if (et == GGUF_TYPE_FLOAT32)
                        {
                            float x;
                            memcpy(&x, buf, 4);
                            kv.arr_value.push_back(std::to_string(x));
                        }
                        else if (et == GGUF_TYPE_FLOAT64)
                        {
                            double x;
                            memcpy(&x, buf, 8);
                            kv.arr_value.push_back(std::to_string(x));
                        }
                        else if (et == GGUF_TYPE_UINT64)
                        {
                            uint64_t x;
                            memcpy(&x, buf, 8);
                            kv.arr_value.push_back(std::to_string(x));
                        }
                        else if (et == GGUF_TYPE_INT64)
                        {
                            int64_t x;
                            memcpy(&x, buf, 8);
                            kv.arr_value.push_back(std::to_string(x));
                        }
                        else
                        {
                            // 逐字节按小端拼成无符号整数
                            uint64_t x = 0;
                            for (int k = sz - 1; k >= 0; --k)
                                x = (x << 8) | buf[k];
                            kv.arr_value.push_back(std::to_string(x));
                        }
                    }
                }
            }
            else if (vt == GGUF_TYPE_STRING)
            {
                if (!read_string(file, pos, kv.value, err, "kv string", file_size))
                    return false;
            }
            else
            {
                int sz = scalar_size((gguf_type)vt);
                if (sz <= 0)
                {
                    err = "unsupported kv value type " + std::to_string(vt);
                    return false;
                }
                if (pos + (uint64_t)sz > file_size)
                {
                    err = "kv data out of bounds";
                    return false;
                }
                std::string raw(sz, '\0');
                if (!file.read_at(pos, &raw[0], sz))
                {
                    err = "unexpected EOF reading kv value";
                    return false;
                }
                pos += sz;
                // 二进制值转可读字符串（f32/f64 直接读成数值）
                if (vt == GGUF_TYPE_FLOAT32)
                {
                    float x;
                    memcpy(&x, raw.data(), 4);
                    kv.value = std::to_string(x);
                }
                else if (vt == GGUF_TYPE_FLOAT64)
                {
                    double x;
                    memcpy(&x, raw.data(), 8);
                    kv.value = std::to_string(x);
                }
                else if (vt == GGUF_TYPE_UINT64)
                {
                    uint64_t x;
                    memcpy(&x, raw.data(), 8);
                    kv.value = std::to_string(x);
                }
                else if (vt == GGUF_TYPE_INT64)
                {
                    int64_t x;
                    memcpy(&x, raw.data(), 8);
                    kv.value = std::to_string(x);
                }
                else
                {
                    uint64_t x = 0;
                    for (int k = sz - 1; k >= 0; --k)
                        x = (x << 8) | (uint8_t)raw[k];
                    kv.value = std::to_string(x);
                }
            }
            info.kv.push_back(std::move(kv));
        }
        return true;
    }

    // 读取张量信息段（对应 gguf.cpp 的 tensor info 循环）
    static bool read_tensor_info(const llama::llama_file &file, uint64_t &pos, int64_t n_tensors, gguf_context &info,
                                 std::string &err, uint64_t file_size)
    {
        for (int64_t i = 0; i < n_tensors; ++i)
        {
            gguf_tensor_info t;
            if (!read_string(file, pos, t.name, err, "tensor name", file_size))
                return false;

            // 张量名不能太长（上游 GGML_MAX_NAME）
            if (t.name.size() >= GGML_MAX_NAME)
            {
                err = "tensor name too long: '" + t.name + "'";
                return false;
            }

            // 张量名不能重复
            for (const auto &e : info.info)
            {
                if (e.name == t.name)
                {
                    err = "duplicate tensor name: '" + t.name + "'";
                    return false;
                }
            }

            // n_dims
            uint32_t n_dims = 0;
            if (!read_le(file, pos, n_dims))
            {
                err = "unexpected EOF reading n_dims";
                return false;
            }
            if (n_dims > GGML_MAX_DIMS)
            {
                err = "tensor '" + t.name + "' n_dims " + std::to_string(n_dims) + " > " + std::to_string(GGML_MAX_DIMS);
                return false;
            }
            t.n_dims = n_dims;

            // ne[]
            for (int32_t j = 0; j < (int32_t)n_dims; ++j)
            {
                if (!read_le(file, pos, t.ne[j]))
                {
                    err = "unexpected EOF reading ne";
                    return false;
                }
                if (t.ne[j] < 0)
                {
                    err = "tensor '" + t.name + "' ne[" + std::to_string(j) + "] < 0";
                    return false;
                }
            }

            // type
            int32_t ty = 0;
            if (!read_le(file, pos, ty))
            {
                err = "unexpected EOF reading tensor type";
                return false;
            }
            t.type = ty;

            // offset
            if (!read_le(file, pos, t.offset))
            {
                err = "unexpected EOF reading tensor offset";
                return false;
            }

            // ---- 派生字节数 & 校验（对照 gguf.cpp 的 gguf_get_tensor_size、gguf_init_from_file） ----
            const type_trait &tr = type_traits((int)t.type);
            if (tr.blck_size == 0 || tr.type_size == 0)
            {
                err = "tensor '" + t.name + "' unsupported/unhandled type " + std::to_string(t.type);
                return false;
            }

            // 行元素数必须是 block size 的整数倍（上游 gguf.c 715 行同样检查）
            if (t.ne[0] % tr.blck_size != 0)
            {
                err = "tensor '" + t.name + "' ne[0]=" + std::to_string(t.ne[0]) + " not multiple of block size " + std::to_string(tr.blck_size);
                return false;
            }

            // 字节数 = (总元素数 / blck_size) * type_size（未量化类型 blck_size=1，故 = 总元素数 × type_size）
            // 总元素数 = ne[0]*ne[1]*ne[2]*ne[3]
            int64_t total = 1;
            for (int j = 0; j < GGML_MAX_DIMS; ++j)
                total *= t.ne[j];
            if (total / tr.blck_size > INT64_MAX / tr.type_size)
            {
                err = "tensor '" + t.name + "' nbytes overflow";
                return false;
            }
            t.nbytes = (total / tr.blck_size) * tr.type_size;

            info.info.push_back(std::move(t));
        }
        return true;
    }

    bool gguf_load(const llama::llama_file &file, gguf_context &info, std::string &err)
    {
        // file 由上层打开后传入：只解析，不再自己开文件（04/05 章线性化）。
        if (!file.valid)
        {
            err = "file not open (valid=false)";
            return false;
        }
        info.file_size = (uint64_t)file.size;

        uint64_t pos = 0;

        // ---- 1. header（对应 gguf.cpp：magic -> version -> n_tensors -> n_kv） ----
        char magic[4] = {0};
        if (!file.read_at(pos, magic, 4))
        {
            err = "unexpected EOF reading magic";
            return false;
        }
        pos += 4;
        if (memcmp(magic, GGUF_MAGIC, 4) != 0)
        {
            err = "bad magic (not a GGUF file)";
            return false;
        }

        if (!read_le(file, pos, info.version))
        {
            err = "unexpected EOF reading version";
            return false;
        }
        if (info.version > GGUF_VERSION)
        {
            err = "unsupported GGUF version " + std::to_string(info.version) + " (max " + std::to_string(GGUF_VERSION) + ")";
            return false;
        }
        if (!read_le(file, pos, info.n_tensors))
        {
            err = "unexpected EOF reading n_tensors";
            return false;
        }
        if (!read_le(file, pos, info.n_kv))
        {
            err = "unexpected EOF reading n_kv";
            return false;
        }
        if (info.n_tensors < 0 || info.n_kv < 0)
        {
            err = "negative n_tensors/n_kv";
            return false;
        }

        // ---- 2. KV 段 ----
        if (!read_kv_pairs(file, pos, info.n_kv, info, err, info.file_size))
        {
            return false;
        }

        // 对齐值：默认 32，若 KV 里有 general.alignment(u32) 则用它（对照 gguf.cpp 613-620）
        for (const auto &kv : info.kv)
        {
            if (kv.key == "general.alignment")
            {
                info.alignment = (uint32_t)strtoull(kv.value.c_str(), nullptr, 10);
                break;
            }
        }
        if (info.alignment == 0 || (info.alignment & (info.alignment - 1)) != 0)
        {
            err = "alignment " + std::to_string(info.alignment) + " is not a power of 2";
            return false;
        }

        // ---- 3. tensor 信息段 ----
        if (!read_tensor_info(file, pos, info.n_tensors, info, err, info.file_size))
        {
            return false;
        }

        // ---- 4. 数据段起点：向上对齐（对应 gguf.cpp 756 行的 GGML_PAD） ----
        info.offset = (info.n_tensors > 0) ? pad(pos, info.alignment) : pos;
        if (info.offset > info.file_size)
        {
            err = "data offset beyond file size";
            return false;
        }

        // ---- 5. 数据一致性 check：每个张量的数据都在文件界内 ----
        // 对应 llama-model-loader.h：offs = data_offset + tensor_offset，须 offs + nbytes <= file_size。
        for (const auto &t : info.info)
        {
            uint64_t offs = info.offset + t.offset;
            if (offs + (uint64_t)t.nbytes < offs || offs + (uint64_t)t.nbytes > info.file_size)
            {
                err = "tensor '" + t.name + "' data not within file bounds, model corrupted or incomplete";
                return false;
            }
        }

        return true;
    }

} // namespace gguf
