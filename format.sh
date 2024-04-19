#!/bin/bash

# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2023.02.19

# Code format script

# Python, yapf version 0.32.0
find ./ -path "./runtime" -prune -o \
    -iname "*.py" -print | xargs yapf -i --style google

# C++ runtime clang-format 16.0.0
find ./runtime/asr_rt/ -iname "*.h" -o \
    -iname "*.cc" -o -iname "*.cpp" | xargs clang-format -style=Google -i

# CMakes cmake-format version 0.6.13
find ./runtime/cmake/ -iname "*.cmake" | xargs cmake-format -i
find ./runtime/ -path "./runtime/3rd_party" -prune -o \
    -path "./runtime/build" -prune -o \
    -iname "CMakeLists.txt" -print | xargs cmake-format -i