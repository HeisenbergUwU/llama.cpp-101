// test-llama-model.cpp - 05 章「建 llama_model + 加载权重」的手写测试（读者填充，见 AGENTS.md：
// 退出码非 0=失败）。用 load_model 把 tinybrainbot 加载成 llama_model，校验 holds 110 个 tensor 且 data 已挂 mmap。

#include "llama-model.h"

#include <cstdio>
#include <cstdlib>

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

    llama::llama_model llm;
    std::string err;

    // TODO(读者)：load_model 实现后应断言——load_model 返回 true；llm.tensors.size()==110
    // （tinybrainbot）；每个 data!=NULL（零拷贝已挂载）；tensors[0]->name 与 GGUF 首 tensor 名一致。
    if (!llama::load_model(argv[1], llm, err))
    {
        return fail(err.c_str());
    }

    std::printf("tensors = %zu\n", llm.tensors.size());
    for (size_t i = 0; i < llm.tensors.size(); i++)
    {
        std::printf("  [%zu] %s  data=%p\n",
                    i,
                    llm.tensors[i]->name,
                    llm.tensors[i]->data);
    }

    return 0;
}
