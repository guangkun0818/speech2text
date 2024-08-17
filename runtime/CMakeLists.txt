# Author: guangkun0818 Email: 609946862@qq.com Created on 2023.08.17

cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(
  asr_rt
  LANGUAGES CXX C
  VERSION 0.1)

# LibTorch requires C++14 at least
set(CMAKE_CXX_STANDARD 17)
link_libraries(stdc++fs) # for std::filesystem if using gcc 7.5
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

include(CTest)
include(FetchContent)

set(FETCHCONTENT_QUIET off)
set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rd_party")
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

include(flashlight_text) # LexiconDecoder
include(gflags)
include(glog)
include(googletest)
include(libtorch)

include_directories(asr_rt)
add_subdirectory(asr_rt)
