# 01 章构建：只依赖标准库，无需链接 ggml
# 用法：
#   make            # 构建 test-load-gguf
#   make run        # 运行（默认测试 ../resources/tinybrainbot-100m-v3-instruct/tinybrainbot-100m-v3-instruct-f16.gguf）
#   make clean

CXX      ?= g++
CXXFLAGS ?= -std=c++11 -Wall -Wextra -O2 -Iinclude
TARGET    = test-load-gguf
MODEL     = ../resources/tinybrainbot-100m-v3-instruct/tinybrainbot-100m-v3-instruct-f16.gguf

all: $(TARGET)

$(TARGET): src/gguf.cpp tests/test-load-gguf.cpp include/gguf.h
	$(CXX) $(CXXFLAGS) src/gguf.cpp tests/test-load-gguf.cpp -o $(TARGET)

run: $(TARGET)
	./$(TARGET) $(MODEL)

clean:
	rm -f $(TARGET)
