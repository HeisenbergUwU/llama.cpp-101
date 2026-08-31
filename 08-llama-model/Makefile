CXX ?= g++
CXXFLAGS ?= -std=c++11 -Wall -Wextra -O3
# 07 章自包含：01(gguf) / 03(ggml) / 04(llama-io) / 05(llama-model) 已拷进本目录
INCLUDES = -Iinclude

TARGET = test-llama-model-vocab
MODEL  = ../resources/tinybrainbot-100m-v3-instruct/tinybrainbot-100m-v3-instruct-f16.gguf

# 07 章自己的源码（llama-hparams + llama-vocab） + 拷进来的前几章源码
SRC = src/llama-hparams.cpp src/llama-vocab.cpp src/llama-model.cpp src/gguf.cpp src/ggml.cpp src/llama-io.cpp

all: $(TARGET)

$(TARGET): $(SRC) tests/test-llama-model-vocab.cpp include/*.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
		$(SRC) tests/test-llama-model-vocab.cpp -o $(TARGET)

run: $(TARGET)
	./$(TARGET) $(MODEL)

clean:
	rm -f $(TARGET)
