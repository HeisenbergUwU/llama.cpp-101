// gguf.h - GGUF 文件裸解析器（load + check）。教程 01 章自包含：只解析 GGUF 元数据并做一致性
// 校验，不依赖 ggml、不 mmap 权重。64 位小端、GGUF 固定小端直接 memcpy 读；04/05 复用本文件。

#pragma once

#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "llama-io.h" // 04 章：llama_file（gguf_load 复用上层打开的文件，不再自己开）

// ---- 常量（取自 gguf.h / ggml.h） ----
#define GGUF_MAGIC "GGUF"
#define GGUF_VERSION 3
#define GGUF_DEFAULT_ALIGNMENT 32
#define GGML_MAX_DIMS 4
#define GGML_MAX_NAME 64

// GGML TYPE 枚举。本项目 tinybrainbot 只用未量化的 F32/F16 两种，
// 真正的 llama.cpp 里有几十种类型。
enum ggml_type_min
{
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
};

enum gguf_type
{
    GGUF_TYPE_UINT8 = 0,
    GGUF_TYPE_INT8 = 1,
    GGUF_TYPE_UINT16 = 2,
    GGUF_TYPE_INT16 = 3,
    GGUF_TYPE_UINT32 = 4,
    GGUF_TYPE_INT32 = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL = 7,
    GGUF_TYPE_STRING = 8,
    GGUF_TYPE_ARRAY = 9,
    GGUF_TYPE_UINT64 = 10,
    GGUF_TYPE_INT64 = 11,
    GGUF_TYPE_FLOAT64 = 12,
    GGUF_TYPE_COUNT, // marks the end of the enum
};

namespace gguf
{

    // ---- 每种张量类型的形状参数（对应 ggml.h 的 ggml_type_traits 表）----
    // 上游每行有 blck_size/type_size（ggml.h 2890-2898）；未量化 blck_size 恒 1，故 type_size=每元素字节数。
    struct type_trait
    {
        std::string name;
        int64_t blck_size; // elements per block
        int64_t type_size; // bytes per block
    };

    // 未列出的类型返回 {unknown, 0, 0}，check 会判为不支持。
    inline const type_trait &type_traits(int t)
    {
        static const std::map<int, type_trait> table = {
            {GGML_TYPE_F32, {"f32", 1, 4}},
            {GGML_TYPE_F16, {"f16", 1, 2}},
        };
        static const type_trait unknown = {"unknown", 0, 0};
        auto it = table.find(t);
        return it == table.end() ? unknown : it->second;
    }

    // ---- 解析结果的中间结构（与 ggml 无关，02 章再用它造 ggml_tensor） ----

    // 一个 KV 键值对（对应上游 ggml/src/gguf.cpp 的 struct gguf_kv）
    struct gguf_kv
    {
        std::string key;
        int64_t type;                       // gguf_type
        std::string value;                  // 标量/单个字符串
        int64_t arr_type = 0;               // 若是 ARRAY，这里是元素类型
        std::vector<std::string> arr_value; // 若是 ARRAY，这里是每个元素
        bool is_array() const { return type == gguf_type::GGUF_TYPE_ARRAY; }
    };

    // 一个张量的「描述」（只有元数据，无数据；对应上游 struct gguf_tensor_info。上游内嵌
    // ggml_tensor 作载体，教程不依赖 ggml，改用等价的 ggml-free 字段）
    struct gguf_tensor_info
    {
        std::string name;
        int64_t n_dims = 0;
        int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1}; // 每维大小，未填=1
        int64_t type = -1;                        // ggml_type_min
        uint64_t offset = 0;                      // 数据在 blob 里的偏移
        // 派生的字节数（check 时算出来）
        int64_t nbytes = 0;
    };

    // 整个 GGUF 文件的解析结果（对应上游 ggml/src/gguf.cpp 的 struct gguf_context；
    // 上游的 offset=数据段起点、size=数据段字节数，这里额外保留 file_size 供 bounds 检查）
    struct gguf_context
    {
        uint32_t version = 0;
        int64_t n_tensors = 0;
        int64_t n_kv = 0;
        uint32_t alignment = GGUF_DEFAULT_ALIGNMENT;
        uint64_t offset = 0;    // 数据段起点（对齐后；对应上游 gguf_context::offset）
        uint64_t file_size = 0; // 整个文件大小（教程自加，供 bounds 校验）

        std::vector<gguf_kv> kv;            // 对应上游 gguf_context::kv
        std::vector<gguf_tensor_info> info; // 对应上游 gguf_context::info
    };

    // ---- 解析入口 ---- 传入已由上层打开的 llama_file，只解析不再自己开文件（04/05 线性化：
    // 文件只开一次，同一份 file 既喂 gguf_load 又给 mmap）。成功 true 填 info，失败 false。
    bool gguf_load(const llama::llama_file &file, gguf_context &info, std::string &err);

    // 把 KV 值转成可读字符串（标量/数组），供打印使用。
    std::string fmt_value(const gguf_kv &kv);

}
