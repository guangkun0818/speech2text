// Author: guangkun0818
// Email: 609946862@qq.com
// Created on 2023.02.25
// Unittest of Flashlight-text decoder.

#include "flashlight/lib/text/decoder/Decoder.h"
#include "flashlight/lib/text/decoder/LexiconDecoder.h"
#include "flashlight/lib/text/decoder/LexiconFreeDecoder.h"
#include "flashlight/lib/text/decoder/Trie.h"
#include "flashlight/lib/text/decoder/lm/KenLM.h"
#include "flashlight/lib/text/dictionary/Defines.h"
#include "flashlight/lib/text/dictionary/Dictionary.h"
#include "flashlight/lib/text/dictionary/Utils.h"
#include "flashlight/lib/text/test/Filesystem.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "torch/torch.h"