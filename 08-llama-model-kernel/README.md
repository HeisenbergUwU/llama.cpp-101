# 08 - 完整前向（全量重算）→ logits + argmax

> 把 06 的「有语义的模型」真正跑起来：输入一批 token（`ForwardInput`），
> 一路算到 `logits[n_tokens × n_vocab]`，并对最后一个 token 做 argmax 得到下一个 token。
> 这是 llama.cpp「解码」的核心——`graph<false>`（并行处理批内所有 token，无 KV cache 增量）。

## 做什么

本章实现**完整的解码器前向**（Scheme A：直接循环，不建计算图）。流程分 6 段：

```
tokens ──▶ ① embed（token_embd 查表）─▶ [N×768]
             │
             ▼ × 12 层（每层两个残差子层）
       ② 注意力：RMSNorm → QKV → RoPE → GQA 注意力 → wo → 残差1
       ③ 前馈：RMSNorm → SwiGLU-PAR（gate‖up→silu⊙→down）→ 残差2
             ▼
       ④ 输出 RMSNorm（output_norm）─▶ ⑤ lm_head（权重绑定，复用 token_embd）
             ▼
       logits[N × 32000] ─▶ ⑥ argmax（末 token）→ 下一个 token id
```

- **① embed**：`token_embd [768,32000]` 里按 token id 取一行（查表）当该 token 的 hidden。
- **② 注意力子层**（每个 transformer 层内）：
  - `RMSNorm(cur, attn_norm)`；
  - QKV 投影：`q=matmul(h,wq) [768]`，`k=matmul(h,wk) [256]`，`v=matmul(h,wv) [256]`；
  - 重排 + RoPE（NEOX/interleaved，只转前 `n_rot=64` 维；**v 不转**）；
  - **GQA 注意力**：`n_head=12` 个 Q 头、`n_head_kv=4` 个 K/V 头，每 `n_gqa=3` 个 Q 头共享一个 K/V 头（省 KV 显存）；
  - `wo` 投影 + 残差 1。
- **③ 前馈子层**：`RMSNorm(ffn)` → **SwiGLU-PAR**：`gate` 与 `up` **各自看原始输入**再乘
  （`silu(gate)·up`，而非串行 `W₂·silu(W₁·x)`）→ `down [768]` + 残差 2。
- **④⑤⑥**：输出 `RMSNorm(output_norm)` → **tied lm_head**（`Model::output == token_embd`，
  复用同一张查表）→ 每 token 一行 logits；对末 token 做 argmax 得下一个 token。

> 层内数据流 / 形状 / GQA 分组 / 残差结构，逐层对照见 `06-llama-model-assembly/model-graph.md`。

### 实测输出

提示 `"The capital of France is"`：

```
末 token argmax id=7831  text=[ Paris]  logit=8.2638
top-3 候选：[7831 ' Paris' 8.264] [276 ' the' 6.280] [31957 '?' 6.006]
```

模型正确预测出 ` Paris` —— 说明 6 段前向的每一步数值都对上了。

## F16 的取舍：为什么「按行反量化」（option c）

tinybrainbot 85 个 matmul 权重是 **F16**，直接 `float` 乘不起来，必须先转 F32。有几种做法：

| 方案 | 做法 | 内存 | 结论 |
|------|------|------|------|
| (a) 全量预转 | 加载时把整张权重转 F32 存 `Model` | 200MB → 400MB+，8GB 机器吃力 | ✗ 浪费 |
| (b) 每次全转 | 每次 matmul 把整张权重全转 F32 | 临时翻倍 | ✗ 重复 |
| **(c) 按行转** | **matmul 一次只反量化一行，进一个复用 scratch 缓冲，F32 累加** | 只多一个 `[2048]` 缓冲 | ✓ |

选 (c) 的原因：**对齐上游 ggml 的按行 `to_float` 语义**（`ggml-cpu` 对每个 block 逐块 `dequantize_row` 也就用到一行/一块），同时**避免在 8GB 机器上为 200MB 权重再造一份 F32 副本**。
整章只有一个函数懂 F16 的字节布局——`dequant_row`（分支 `W->type`：F16 → 逐元素 `fp16_to_fp32`；F32 → `memcpy`）。所有算子统一吃裸 `float*`，天然和"权重是 F16 还是 F32"解耦，这正是 13+ 章逐算子深入能**原位改**的地方。

要的 `fp16_to_fp32` 是手写的 IEEE 754 半精度→单精度（含 0/非规格化/无穷/NaN 分支），无三方依赖。

## 三个接缝（seam）：为 11 章 KV cache 预留

11 章会插入真正的 KV cache（自回归时只喂新 token、复用历史 K/V）。为了**不重写本章算子**，
先把三个"改刀点"定好、并把"前向内部如何访问历史 K/V"抽象出来：

| 接缝 | 声明 | 08 章行为 | 11 章改动 |
|------|------|-----------|-----------|
| **1. `KVSlice`** | 所有层历史 K/V 的不透明视图，只暴露 `n_kv` | 每次 `forward()` **全量重算**并填满（`n_kv = N`） | 保留旧 K/V，只 append 新增 token |
| **2. `build_causal_mask(n)`** | 独立函数，返回 `[n×n]` 矩阵 | `s>t` 置 `-inf` | 改成 `[n_kv×n_tokens]` |
| **3. `forward(...)`** | 唯一入口（固定签名） | `kv.n_kv = N` 后一起算 | 自回归 loop 逐 token，从 KVSlice 读历史 |

关键约定：**注意力只读 `KVSlice::n_kv`，绝不假定它等于当前 token 数**。所以 08 章把"本批
N 个 token 的 K/V 填进 KVSlice"这个动作刻意藏在 `forward()` 内部（就是 11 章"append K/V"
的落刀点），`forward()` 之外看不到任何 K/V 布局细节。

## kernel 层：算子签名（裸 `float*`，13 章+ 原位深入）

所有算子放进独立的 `namespace kernel`，与 `llama`（前向/模型语义）分开：

```cpp
namespace kernel {
    float fp16_to_fp32(uint16_t h);                                    // 纯标量原语
    bool  check_shape(const ggml::ggml_tensor* t, int n_dims, const int64_t* need_ne); // 共享维度校验
    void  dequant_row(const ggml::ggml_tensor* W, int row, float* dst); // 唯一懂 F16 的函数
    bool  rms_norm(const ggml::ggml_tensor* x, const ggml::ggml_tensor* W, float eps, ggml::ggml_tensor* out);
    bool  matmul(const ggml::ggml_tensor* x, const ggml::ggml_tensor* W, float* scratch, ggml::ggml_tensor* out);
    void  silu(float* x, int n);                 // in place
    void  rope_inplace(float* x, int n_rot, int pos, float base); // NEOX 对间旋转
    void  softmax_row(float* x, int n);          // in place 单行
}
```

算子都是「无状态并且输入输出是一段 `float*` 或一行 `ggml::ggml_tensor`」：**凡带权重的算子
（`dequant_row`/`matmul`/`rms_norm`）现在把激活行 x/out 包成一行 `ggml::ggml_tensor`
（`make_row`，见 `llama-forward.cpp`），并在函数内用 `check_shape` 校验尺寸**
（如 `matmul` 断 `W->ne[0]==x->ne[0]`、`W->ne[1]==out->ne[0]`；`rms_norm` 断 `W->ne[0]==x->ne[0]==out->ne[0]`），
不符返回 `false`。`rms_norm`/`matmul` 因此返回 `bool`；`silu`/`rope_inplace`/`softmax_row`
是单行逐元素算子、无权重可比对，保持 `float*`。临时激活的 `data` 仍是 caller 的 flat `float*`
缓冲（view 不拥有内存），这正是「直接循环、不建图」的体现。
13 章起想深挖某个算子（向量化、低精度、分块），只需改这一个函数 + `test-kernel.cpp`
里的单测——不动 `forward()` 的外围接线。这是把 kernel 单独拎出来、可独立编译测试的原因。

## 怎么跑（CMake，两个 CMakeLists）

本章拆成 **kernel（纯算子）** 与 **具体组件（forward/模型胶水）** 两层，**各有独立的
CMakeLists.txt**：`src/kernel/CMakeLists.txt` 可单独构建 kernel + 它的测试；根
`CMakeLists.txt` 用 `add_subdirectory(src/kernel)` 引入再搭整体前向。

```bash
cd 08-llama-model-kernel

# --- kernel 单独构建（不依赖模型 / 外层组件） ---
cmake -B build-kernel -S src/kernel
cmake --build build-kernel
./build-kernel/test-kernel            # 只验算子，无需模型权重

# --- 整体构建（根 CMakeLists 引入 kernel） ---
cmake -B build -S .
cmake --build build

# kernel 测试（在 build/src/kernel/ 下，随整体验证用）
./build/src/kernel/test-kernel
# 整体前向：加载 tinybrainbot -> logits -> argmax
./build/test-llama-forward ../resources/...-f16.gguf
# 或用便捷目标
cmake --build build --target run-forward
```

- **kernel**（`src/kernel/CMakeLists.txt`）= 7 个无状态算子，`libkernel.a` 不链接模型栈，
  所以能**单独建一个 build-kernel**、只验算子；
- **根**（`CMakeLists.txt`）= `add_subdirectory(src/kernel)` + 外层 `llama-forward`
  （KVSlice / build_causal_mask / forward + 模型栈），`test-llama-forward` 用真实模型
  跑到 **PASS**（退出码 0）并打印 argmax 候选。

`-std=c++11 -Wall -Wextra -O3`，编译干净、无警告。

> 注：clangd 对 `gguf.h`/`ggml.h`/`llama-io.h` 报"找不到头文件"是环境假阳性（未配各章 `-Iinclude`），以 `cmake`/编译器为准（AGENTS.md）。

## 关键文件

| 文件 | 作用 |
|------|------|
| `include/kernel/kernel.h` | kernel 层：7 个算子声明（`namespace kernel`，只依赖 ggml） |
| `src/kernel/cpu/kernel.cpp` | kernel 层：7 个算子实现（可单独编译/测试） |
| `include/llama-forward.h` | 组件层：`ForwardInput` / `KVSlice` / `build_causal_mask` / `forward` 声明 |
| `src/llama-forward.cpp` | 组件层：6 段前向主流程（调用 kernel 算子） |
| `tests/test-kernel.cpp` | kernel 单独测试（无需模型，`libkernel.a` 即可） |
| `tests/test-llama-forward.cpp` | 整体前向测试（加载真实模型） |
| `CMakeLists.txt` | 分层构建：`kernel` / `llama-forward` 库 + 两个测试目标 |

## 对照上游

- `llama.cpp/src/models/llama.cpp` —— `graph<false>`（解码器主前向：residual 两处 ADD、
  `build_inp_embd` 查表、`LLM_FFN_SILU`+`LLM_FFN_PAR` 的 SwiGLU 接法）
- `llama.cpp/src/llama-graph.cpp` —— `build_norm`（RMSNorm）、`build_qkv`、`build_attn_mha`
  （GQA 分组）、`build_ffn`、`build_inp_embd`、`build_causal_mask`
- `llama.cpp/ggml/src/ggml-cpu/ops.cpp` —— `rms_norm_f32`、`soft_max_f32`、`rope_f32` 内核
- `llama.cpp/src/llama-vocab.h` —— `tokenize`（SPM）

> 更精简对照见 `reference.md`。

---

下一章进度见 `ROADMAP.md`（09 自回归采样：拿到 logits 后 greedy 逐 token 生成，处理 EOG/max_tokens）。
