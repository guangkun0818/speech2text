# Author: guangkun0818 Email: 609946862@qq.com Created on 2023.08.17

# Flashlight-Text, LexiconDecoder impl
FetchContent_Declare(
  flashlight_text
  GIT_REPOSITORY https://github.com/flashlight/text.git
  GIT_TAG v0.0.2)
FetchContent_MakeAvailable(flashlight_text)

include_directories(${flashlight_text_SOURCE_DIR})
