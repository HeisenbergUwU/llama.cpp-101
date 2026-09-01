# src/kernel/CMakeLists.txt - kernel 子项目（独立可构建）
#
# 定义 kernel 库（纯算子）与它的单独测试 test-kernel。
# 依赖极简：kernel 只 include 08 根的 include/ 拿 "ggml.h"（ggml::ggml_tensor），
# 不链接任何模型栈 —— 所以可以完全独立 cmake + 独立跑测试，验证算子本身。
#
# 单独构建：
#   cmake -B build-kernel -S src/kernel
#   cmake --build build-kernel && ./build-kernel/test-kernel
#
# 也被根 CMakeLists 用 add_subdirectory(src/kernel) 引入（复用同一个 kernel 目标）。

cmake_minimum_required(VERSION 3.16)
project(llama101-kernel CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra)

# kernel.h / ggml.h 都位于 08 根的 include/（本文件在其 src/kernel/ 下，上溯两级）
set(KERNEL_INCLUDE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../include")

# ---------- kernel 库：纯算子，不依赖模型栈 ----------
add_library(kernel STATIC cpu/kernel.cpp)
target_include_directories(kernel PUBLIC ${KERNEL_INCLUDE_ROOT})

# ---------- test-kernel：只测算子（无需模型） ----------
# 测试源码在 08 根的 tests/（上溯两级）
add_executable(test-kernel ../../tests/test-kernel.cpp)
target_link_libraries(test-kernel PRIVATE kernel)

# ---------- 便捷运行目标 ----------
add_custom_target(run-kernel
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test-kernel
    DEPENDS test-kernel)
