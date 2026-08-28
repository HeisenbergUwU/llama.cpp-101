CXX ?= g++
CXXFLAGS ?= -std=c++11 -Wall -Wextra -O3
# 05 章自包含：01(gguf) / 03(ggml) / 04(llama-io) 前几章源码已拷进本目录
INCLUDES = -Iinclude

TARGET = test-llama-model
MODEL  = ../resources/tinybrainbot-100m-v3-instruct/tinybrainbot-100m-v3-instruct-f16.gguf

# 05 章自己的源码 + 拷进来的前几章源码(含 04 的 llama-io 封装)
SRC = src/llama-model.cpp src/gguf.cpp src/ggml.cpp src/llama-io.cpp

all: $(TARGET)

$(TARGET): $(SRC) tests/test-llama-model.cpp include/llama-model.h include/gguf.h include/ggml.h include/llama-io.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
		$(SRC) tests/test-llama-model.cpp -o $(TARGET)

run: $(TARGET)
	./$(TARGET) $(MODEL)

clean:
	rm -f $(TARGET)
