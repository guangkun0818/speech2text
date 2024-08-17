# Author: guangkun0818 Email: 609946862@qq.com Created on 2023.08.17

enable_testing()

# Unitttest of flashlight decoder
add_executable(fl_decoder_test fl_decoder_test.cc)
target_link_libraries(fl_decoder_test gtest_main glog flashlight-text-kenlm
                      ${TORCH_LIBRARIES})

add_test(
  NAME fl_decoder_test
  COMMAND fl_decoder_test
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
