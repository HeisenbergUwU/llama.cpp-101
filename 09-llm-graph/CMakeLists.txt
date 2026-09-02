# CMakeLists.txt - 09 章「ggml 计算图设施」整体构建（根）
#
# 分层（两个独立的 CMakeLists）：
#   src/kernel/CMakeLists.txt —— kernel 纯算子子项目，可单独 cmake/build/test；
#                                本文件用 add_subdirectory(src/kernel) 引入复用其 kernel 目标。
#   本文件                       —— ggml_graph 计算图设施（ggml-graph.cpp）
#                                 + 它的单测 test-ggml-graph（不依赖模型）。
#
# 用法：
#   整体：cmake -B build && cmake --build build
#         ./build/test-ggml-graph
#   kernel 单独：见 src/kernel/CMakeLists.txt（不依赖本文件 / 图）。

cmake_minimum_required(VERSION 3.16)
project(llama101-ch09-graph CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra)

# 引入 kernel 子项目（kernel 库 + test-kernel）——复用其 kernel 目标
add_subdirectory(src/kernel)

# ---------- ggml-graph 库：计算图设施 ----------
# 在 03 章加载层（ggml.cpp 池子/ggml_new_tensor）之上建图，只用 kernel 算子，不依赖模型栈。
add_library(ggml-graph STATIC src/ggml-graph.cpp src/ggml.cpp)
target_include_directories(ggml-graph PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ggml-graph PUBLIC kernel)

# ---------- test-ggml-graph：单测（无需模型） ----------
add_executable(test-ggml-graph tests/test-ggml-graph.cpp)
target_link_libraries(test-ggml-graph PRIVATE ggml-graph)

# ---------- 便捷运行目标 ----------
add_custom_target(run-graph
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test-ggml-graph
    DEPENDS test-ggml-graph)
