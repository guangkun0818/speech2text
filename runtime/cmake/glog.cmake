# Author: guangkun0818 Email: 609946862@qq.com Created on 2023.08.17

# Glog v0.4.0
FetchContent_Declare(
  glog
  GIT_REPOSITORY https://github.com/google/glog.git
  GIT_TAG v0.4.0)
FetchContent_MakeAvailable(glog)

include_directories(${glog_SOURCE_DIR}/src ${glog_BINARY_BIN})
