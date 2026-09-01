# 07 - 词表 Vocab（tokenize / detokenize）

> 从 GGUF 的 `tokenizer.ggml.*` KV 建词表，提供 token id <-> 文本的编解码。

## 做什么

把文本与 token id 打通——这是让 `token_embd` 那个 `[768, 32000]` 张量真正与文字挂钩的一步。

- **`build`**：从 `tokenizer.ggml.tokens/scores/token_type/特殊 id` 建词表（tinybrainbot 是 SPM 型，`model=llama`，`n_vocab=32000`）。
- **`tokenize`（文字 -> id）**：SPM 无 merges → 逐 token 最长前缀匹配（`token_to_id` 命中即取）；**词表外字符按 UTF-8 逐字节编码成 `<0xXX>` byte token**（无损兜底，对齐上游 `resegment` 的 `byte_to_token`）。中文因此可无损往返。
- **`detokenize`（id -> 文字）**：normal token 直接输出，byte token 还原成字节，`▁`（U+2581 SPM 空格）转回空格。
- **`is_eog`**：判断 eos/bos 等句末 token（后面自回归采样章的停止条件）。
- **`llama_model` 集成**：`llama_model` 持有 `Vocab vocab` 成员，`load_model` 内部调用 `vocab.build`。测试通过 `llm.vocab.*` 调用。

> 教学取舍：上游 `llama_vocab` 用 pimpl、走 `llama_model_loader + LLM_KV`、支持几十种分词器；迷你版扁平化、直接读 `gguf_context.kv`、只做 SPM + byte fallback。

## 怎么跑

```bash
cd 07-llama-model-vocab && make run

# 把词表导出成 JSON（每个 token 的 id/text/score/token_type）
python3 tests/export_vocab_json.py ../resources/tinybrainbot-100m-v3-instruct/tinybrainbot-100m-v3-instruct-f16.gguf ./vocab.json
```

## 关键文件

| 文件 | 作用 |
|------|------|
| `include/llama-vocab.h` | `struct Vocab`（tokens + token_to_id 反向表 + 特殊 id + 接口） |
| `src/llama-vocab.cpp` | `build` / `tokenize` / `detokenize` / `is_eog` / `token_to_byte` / `byte_to_token` |
| `tests/test-llama-model-vocab.cpp` | 手写测试：n_vocab / 特殊 id / is_eog / **中文 + 英文 round-trip** |
| `tests/export_vocab_json.py` | Python 脚本：把词表导出成 JSON |

## 对照上游

- `llama.cpp/src/llama-vocab.h/.cpp` —— `llama_vocab`、`tokenize`/`token_to_piece`/`is_eog`/`byte_to_token`

> 更精简对照见 `reference.md`。

---

下一章：**08 - 完整前向**（先从「前向输入结构 ForwardInput」起步，后续逐步写计算；进度见 `ROADMAP.md`）。
