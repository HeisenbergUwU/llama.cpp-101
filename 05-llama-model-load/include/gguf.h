// gguf.h - GGUF 文件裸解析器（load + check）
//
// 这是教程 01 章的自包含实现：只解析 GGUF 文件的元数据（header、KV、tensor 信息）
// 并对它们做一致性校验，不依赖 ggml 的 tensor/context，也不 mmap 权重数据。
//
// 设计原则（与 llama.cpp 上游逻辑一一对应，见参考文献）：
//   - 文件布局  : ggml/include/gguf.h（头部注释的结构定义）
//   - 读取顺序  : ggml/src/gguf.cpp（gguf_init_from_file 的实现）
//   - 越界校验  : src/llama-model-loader.h（llama_tensor_weight 的 bounds 检查）
//
// 命名对齐：本文件里的类型名尽量跟上游 llama.cpp 一致——
//   - gguf_context     对应上游 ggml/src/gguf.cpp 的 struct gguf_context
//   - gguf_tensor_info 对应上游 ggml/src/gguf.cpp 的 struct gguf_tensor_info
//   - gguf_kv          对应上游 ggml/src/gguf.cpp 的 struct gguf_kv
//
// 阶段边界（与 llama.cpp 真实分层对齐）：
//   本文件 = 只读元数据（对应 gguf_context 阶段）
//   02 章   = 再用这些元数据创建 ggml_tensor、映射权重（对应 llama_model_loader）
//   03 章   = 推理
//
// 本机是 64 位小端环境（macOS x86），GGUF 固定小端，所以直接 memcpy 读即可。

#pragma once

#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

// ---- 常量（取自 gguf.h / ggml.h） ----
#define GGUF_MAGIC "GGUF"
#define GGUF_VERSION 3
#define GGUF_DEFAULT_ALIGNMENT 32
#define GGML_MAX_DIMS 4
#define GGML_MAX_NAME 64

// GGML TYPE 的枚举。本项目 tinybrainbot-100m-v3-instruct-f16 只用未量化的
// F32 和 F16 两种，所以这里只列这两个（真正的 llama.cpp 里有几十种类型）。
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

    // ---- 每种张量类型的形状参数（对应 ggml.h 里的 ggml_type_traits 表） ----
    // 上游 schema 每行都有 blck_size 和 type_size 两个字段（见 ggml.h 2890-2898）；
    // 本项目只用到未量化的 F32/F16，它们的 blck_size 恒为 1，所以 type_size 恰好
    // 就等于“每元素字节数”。这里同样保留两个字段，和上游保持一致。
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

    // 一个张量的「描述」（只有元数据，没有数据；对应上游 ggml/src/gguf.cpp 的
    // struct gguf_tensor_info。上游还内嵌了一个 ggml_tensor 作为载体，这里因
    // 教程不依赖 ggml，改用等价的 ggml-free 字段）
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

    // ---- 解析入口 ----
    // 成功返回 true，并把结果填进 info；失败返回 false（out 里记录错误信息）。
    bool gguf_load(const std::string &fname, gguf_context &info, std::string &err);

    // 把 KV 值转成可读字符串（标量/数组），供打印使用。
    std::string fmt_value(const gguf_kv &kv);

}
