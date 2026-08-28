CXX ?= g++
CXXFLAGS ?= -std=c++11 -Wall -Wextra -O3 -Iinclude
TARGET = test-ggml-build-context
MODEL  = ../resources/tinybrainbot-100m-v3-instruct/tinybrainbot-100m-v3-instruct-f16.gguf

all: $(TARGET)

$(TARGET): src/ggml.cpp tests/test-ggml-build-context.cpp include/ggml.h
	$(CXX) $(CXXFLAGS) src/ggml.cpp tests/test-ggml-build-context.cpp -o $(TARGET)

run: $(TARGET)
	./$(TARGET) $(MODEL)

clean:
	rm -f $(TARGET)
