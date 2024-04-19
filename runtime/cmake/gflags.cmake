# Gflags v0.4.0
FetchContent_Declare(
  gflags
  GIT_REPOSITORY https://github.com/gflags/gflags.git
  GIT_TAG v2.2.1)
FetchContent_MakeAvailable(gflags)

include_directories(${glog_BINARY_BIN}/include)
