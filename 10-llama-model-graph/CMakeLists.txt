# CMakeLists.txt - 10 章「整体模型图」整体构建（根）
#
# 分层（两个独立的 CMakeLists）：
#   src/kernel/CMakeLists.txt —— kernel 纯算子子项目，可单独 cmake/build/test；
#                                本文件用 add_subdirectory(src/kernel) 引入复用其 kernel 目标。
#   本文件                       —— ggml 图设施（ggml-graph/ops/kernel）+ 语义模型栈
#                                 + 在本章把 llama 前向建成整图的 llama-graph（build_model_graph）
#                                 + 两个测试：
#                                  · test-ggml-graph（无需模型，验算子/图设施）
#                                  · test-llama-graph-model（需模型，验整图 logits）
#
# 用法：
#   整体：cmake -B build && cmake --build build
#         ./build/test-ggml-graph
#         ./build/test-llama-graph-model <model.gguf>
#   kernel 单独：见 src/kernel/CMakeLists.txt（不依赖本文件 / 图）。

cmake_minimum_required(VERSION 3.16)
project(llama101-ch10-model-graph CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra)

# 引入 kernel 子项目（kernel 库 + test-kernel）——复用其 kernel 目标
add_subdirectory(src/kernel)

# ---------- ggml 库：池子/加载层（03 章，ggml.cpp）----------
# ggml-ops（建节点）与 ggml-kernel（分发）都要调它的函数，独立成库避免循环依赖。
add_library(ggml STATIC src/ggml.cpp)
target_include_directories(ggml PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---------- ggml-kernel 库：图 ↔ 算子的分发/组合层 ----------
# 把「op 值 → 具体算子」的 switch 收进来（src/ggml-kernel.cpp），
# 底层复用 kernel 裸 float* 单算子。计算图设施只依赖本层。
add_library(ggml-kernel STATIC src/ggml-kernel.cpp)
target_include_directories(ggml-kernel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ggml-kernel PUBLIC kernel ggml)

# ---------- ggml-ops 库：算子构造函数（建节点）----------
add_library(ggml-ops STATIC src/ggml-ops.cpp)
target_include_directories(ggml-ops PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ggml-ops PUBLIC ggml)

# ---------- ggml-graph 库：计算图骨架（build/compute）----------
# 只负责「登记 + 按拓扑执行」：建节点走 ggml-ops，执行走 ggml-kernel。
add_library(ggml-graph STATIC src/ggml-graph.cpp)
target_include_directories(ggml-graph PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ggml-graph PUBLIC ggml ggml-ops ggml-kernel)

# ---------- test-ggml-graph：单测（无需模型） ----------
add_executable(test-ggml-graph tests/test-ggml-graph.cpp)
target_link_libraries(test-ggml-graph PRIVATE ggml-graph)

# ---------- llama-model 库：语义模型栈（gguf/io/hparams/model/vocab） ----------
# 在本章自包含拷贝里，把「加载出有语义的模型」的各章源码编进一个库，
# 供 build_model_graph（拿 model.token_embd / layers[il].wq 等权重）与模型测试用。
add_library(llama-model STATIC
    src/gguf.cpp
    src/llama-io.cpp
    src/llama-hparams.cpp
    src/llama-model.cpp
    src/llama-vocab.cpp)
target_include_directories(llama-model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(llama-model PUBLIC ggml)

# ---------- llama-graph 库：整体模型图构建器（build_model_graph） ----------
# 用 10 章算子把 llama 前向（embed→12 层→lm_head）建成一张 ggml_cgraph。
add_library(llama-graph STATIC src/llama-graph.cpp)
target_include_directories(llama-graph PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(llama-graph PUBLIC llama-model ggml-ops ggml-graph)

# ---------- test-llama-graph-model：整体模型图（需模型） ----------
add_executable(test-llama-graph-model tests/test-llama-graph-model.cpp)
target_link_libraries(test-llama-graph-model PRIVATE llama-graph)

# ---------- 便捷运行目标 ----------
add_custom_target(run-graph
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test-ggml-graph
    DEPENDS test-ggml-graph)

# tinybrainbot 默认模型路径（相对本目录，纯参考；README 里用相对路径引 resources）
set(TINYBRAINBOT_MODEL
    "${CMAKE_CURRENT_SOURCE_DIR}/../resources/tinybrainbot-100m-v3-instruct/tinybrainbot-100m-v3-instruct-f16.gguf")
add_custom_target(run-model-graph
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test-llama-graph-model ${TINYBRAINBOT_MODEL}
    DEPENDS test-llama-graph-model)
