CXX ?= g++
CXXFLAGS ?= -std=c++11 -Wall -Wextra -O3
# 04 章:自包含(01 gguf + 03 ggml 已拉进来,gguf 的文件操作已改走 llama-io)
INCLUDES = -Iinclude

MODEL  = ../resources/tinybrainbot-100m-v3-instruct/tinybrainbot-100m-v3-instruct-f16.gguf
BINS   = test-llama-io test-gguf-io

all: $(BINS)

# IO 封装层测试(只测 llama_file / llama_mmap)
test-llama-io: src/llama-io.cpp tests/test-llama-io.cpp include/llama-io.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
		src/llama-io.cpp tests/test-llama-io.cpp -o test-llama-io

# gguf 适配测试(gguf_load 已改走 llama_file,需连 gguf.cpp + llama-io.cpp)
test-gguf-io: src/gguf.cpp src/llama-io.cpp tests/test-gguf-io.cpp include/gguf.h include/llama-io.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
		src/gguf.cpp src/llama-io.cpp tests/test-gguf-io.cpp -o test-gguf-io

run: all
	./test-llama-io $(MODEL)
	./test-gguf-io $(MODEL)

clean:
	rm -f $(BINS)
