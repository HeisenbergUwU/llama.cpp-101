# src/kernel/CMakeLists.txt - kernel 子项目（独立可构建）
#
# 定义 kernel 库（纯算子）。依赖极简：kernel 只 include 本目录的 include/ 拿
# "ggml.h"（ggml::ggml_tensor），不链接任何模型栈 —— 可被根 CMakeLists 用
# add_subdirectory(src/kernel) 引入复用。
#
# 09 章不再单独测 kernel：算子的数值由 test-ggml-graph 的 compute 走一遍覆盖。

cmake_minimum_required(VERSION 3.16)
project(llama101-kernel CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra)

# kernel.h / ggml.h 都位于 09 章的 include/（本文件在其 src/kernel/ 下，上溯两级）
set(KERNEL_INCLUDE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../include")

# ---------- kernel 库：纯算子，不依赖模型栈 ----------
add_library(kernel STATIC cpu/kernel.cpp)
target_include_directories(kernel PUBLIC ${KERNEL_INCLUDE_ROOT})
