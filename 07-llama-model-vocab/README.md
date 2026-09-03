# 07 - 词表 Vocab（tokenize / detokenize）

> 从 GGUF 的 `tokenizer.ggml.*` KV 建词表，提供 token id <-> 文本的编解码。

## 做什么
- `build`：从 `tokenizer.ggml.tokens/scores/token_type/特殊 id` 建词表（tinybrainbot 为 SPM 型，`n_vocab=32000`）
- `tokenize`（文字 -> id）：SPM 无 merges → 逐 token 最长前缀匹配；词表外字符按 UTF-8 逐字节编码成 `<0xXX>` byte token（无损兜底，中文可无损往返）
- `detokenize`（id -> 文字）：normal token 直接输出，byte token 还原字节，`▁`（SPM 空格）转回空格
- `is_eog`：判断 eos/bos 等句末 token（供后续采样章作停止条件）
- `llama_model` 集成：`llama_model` 持有 `Vocab vocab` 成员，`load_model` 内部调用 `vocab.build`
- 教学取舍：上游 `llama_vocab` 用 pimpl + 几十种分词器；迷你版扁平化，只做 SPM + byte fallback

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
| `tests/test-llama-model-vocab.cpp` | 手写测试：n_vocab / 特殊 id / is_eog / 中文 + 英文 round-trip |
| `tests/export_vocab_json.py` | Python 脚本：把词表导出成 JSON |

## 对照上游
- `llama.cpp/src/llama-vocab.h/.cpp` —— `llama_vocab`、`tokenize`/`token_to_piece`/`is_eog`/`byte_to_token`

---
下一章：**08 - 完整前向 / 算子内核**（用 kernel 算子把语义模型跑成 logits）。
