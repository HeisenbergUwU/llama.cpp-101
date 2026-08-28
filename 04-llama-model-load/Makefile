CXX ?= g++
CXXFLAGS ?= -std=c++11 -Wall -Wextra -O3
# 04 章自包含：01(gguf) / 03(ggml) 的前几章源码已直接拷进本目录 src/ 和 include/
INCLUDES = -Iinclude

TARGET = test-llama-model
MODEL  = ../resources/tinybrainbot-100m-v3-instruct/tinybrainbot-100m-v3-instruct-f16.gguf

# 04 章自己的源码 + 拷进来的前几章源码
SRC = src/llama-model.cpp src/gguf.cpp src/ggml.cpp

all: $(TARGET)

$(TARGET): $(SRC) tests/test-llama-model.cpp include/llama-model.h include/gguf.h include/ggml.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
		$(SRC) tests/test-llama-model.cpp -o $(TARGET)

run: $(TARGET)
	./$(TARGET) $(MODEL)

clean:
	rm -f $(TARGET)
