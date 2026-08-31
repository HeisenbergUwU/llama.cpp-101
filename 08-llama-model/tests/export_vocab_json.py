#!/usr/bin/env python3
# export_vocab_json.py - 07 章：把 GGUF 里的词表导出成 JSON
#
# 用法:  python3 export_vocab_json.py <model.gguf> [out.json]
#   - 只解析 GGUF 元数据里的 tokenizer.ggml.* KV（不 mmap 权重），
#   - 把每个 token 的 id/text/score/token_type 导出成 JSON，
#   - 不传 out.json 时打印到 stdout。
#
# GGUF 类型码(见 07 include/gguf.h 的 enum gguf_type):
#   0 UINT8 1 INT8 2 UINT16 3 INT16 4 UINT32 5 INT32 6 FLOAT32
#   7 BOOL 8 STRING 9 ARRAY 10 UINT64 11 INT64 12 FLOAT64

import json
import os
import struct
import sys

TYPE_NAMES = {
    0: "UNDEFINED", 1: "NORMAL", 2: "UNKNOWN", 3: "CONTROL",
    4: "USER_DEFINED", 5: "UNUSED", 6: "BYTE",
}


def u32(f):
    return struct.unpack("<I", f.read(4))[0]


def u64(f):
    return struct.unpack("<Q", f.read(8))[0]


def read_str(f):
    n = u64(f)
    return f.read(n).decode("utf-8", "replace")


def read_val(f, t):
    """读一个 scalar 或 string (不含 ARRAY)"""
    if t == 0:
        return struct.unpack("<B", f.read(1))[0]
    if t == 1:
        return struct.unpack("<b", f.read(1))[0]
    if t == 2:
        return struct.unpack("<H", f.read(2))[0]
    if t == 3:
        return struct.unpack("<h", f.read(2))[0]
    if t == 4:
        return struct.unpack("<I", f.read(4))[0]
    if t == 5:
        return struct.unpack("<i", f.read(4))[0]
    if t == 6:
        return struct.unpack("<f", f.read(4))[0]
    if t == 7:
        return struct.unpack("<B", f.read(1))[0]
    if t == 8:
        return read_str(f)
    if t == 10:
        return u64(f)
    if t == 11:
        return struct.unpack("<q", f.read(8))[0]
    if t == 12:
        return struct.unpack("<d", f.read(8))[0]
    raise ValueError("unknown scalar type %d" % t)


def read_array(f, elem_type):
    n = u64(f)
    return [read_val(f, elem_type) for _ in range(n)]


def load_kv(path):
    kv = {}
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError("not a GGUF file: %r" % magic)
        u32(f)  # version
        u64(f)  # n_tensors (跳过，只要 KV)
        n_kv = u64(f)
        for _ in range(n_kv):
            key = read_str(f)
            t = u32(f)
            if t == 9:  # ARRAY: 元素类型 + 个数 + 各元素
                arr_type = u32(f)
                kv[key] = read_array(f, arr_type)
            else:
                kv[key] = read_val(f, t)
    return kv


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("用法: %s <model.gguf> [out.json]\n" % sys.argv[0])
        return 2

    kv = load_kv(sys.argv[1])

    tokens = kv.get("tokenizer.ggml.tokens", [])
    scores = kv.get("tokenizer.ggml.scores", [])
    types = kv.get("tokenizer.ggml.token_type", [])

    vocab = {
        "model": kv.get("tokenizer.ggml.model", ""),
        "pre": kv.get("tokenizer.ggml.pre", ""),
        "n_vocab": len(tokens),
        "tokens": [],
    }
    for i, text in enumerate(tokens):
        score = scores[i] if i < len(scores) else 0.0
        t = types[i] if i < len(types) else 1
        vocab["tokens"].append({
            "id": i,
            "text": text,
            "score": score,
            "token_type": t,
            "type_name": TYPE_NAMES.get(t, "?"),
        })

    out = json.dumps(vocab, ensure_ascii=False, indent=1)
    if len(sys.argv) >= 3:
        # 若输出目标是目录，自动在其中拼一个默认文件名（<gguf名>_vocab.json）
        out_path = sys.argv[2]
        if out_path == "." or os.path.isdir(out_path):
            base = os.path.splitext(os.path.basename(sys.argv[1]))[0]
            out_path = os.path.join(out_path, base + "_vocab.json")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(out + "\n")
        sys.stderr.write("已写入 %s (共 %d 个 token)\n" % (out_path, len(tokens)))
    else:
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
