# CMakeLists.txt - 08 章「完整前向」整体构建（根）
#
# 分层（两个独立的 CMakeLists）：
#   src/kernel/CMakeLists.txt —— kernel 纯算子子项目，可单独 cmake/build/test；
#                                本文件用 add_subdirectory(src/kernel) 引入复用其 kernel 目标。
#   本文件                       —— 外层组件：llama-forward（KVSlice / build_causal_mask / forward
#                                 + 模型栈 gguf/ggml/llama-io/hparams/model/vocab）+ test-llama-forward。
#
# 用法：
#   整体：cmake -B build && cmake --build build
#         ./build/test-llama-forward resources/...-f16.gguf
#   kernel 单独：见 src/kernel/CMakeLists.txt（不依赖本文件 / 模型）。

cmake_minimum_required(VERSION 3.16)
project(llama101-ch08-kernel CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra)

# 引入 kernel 子项目（kernel 库 + test-kernel）——复用其 kernel 目标
add_subdirectory(src/kernel)

# tinybrainbot 默认模型路径（相对本目录）
set(TINYBRAINBOT_MODEL
    "${CMAKE_CURRENT_SOURCE_DIR}/../resources/tinybrainbot-100m-v3-instruct/tinybrainbot-100m-v3-instruct-f16.gguf")

# ---------- llama-forward 库：具体组件 + 模型栈 ----------
# 模型栈各章拷贝的源码：01(gguf) / 03(ggml) / 04(llama-io) / 06(hparams) / 07(vocab,model)
add_library(llama-forward STATIC
    src/llama-forward.cpp
    src/gguf.cpp
    src/ggml.cpp
    src/llama-io.cpp
    src/llama-hparams.cpp
    src/llama-model.cpp
    src/llama-vocab.cpp)
target_include_directories(llama-forward PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(llama-forward PUBLIC kernel)
