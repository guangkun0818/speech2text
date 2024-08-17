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

using namespace fl::lib::text;

// The token dictionary for this test defines this separator token.
// If seperate token has already tokenized into tokens, subword specificly,
// kSepToken should be "<blank_id>" which will be squeezed out during decoding.
constexpr const char* kSepToken = "|";

// Set sample data dir from CMake
static fs::path sample_data_dir = "test_data";

class TestFlashlightDecoder : public ::testing::Test {
 protected:
  void SetUp() override {
    // Load lexicon.txt
    fs::path word_lst = sample_data_dir / "words.lst";
    lexicon_ = loadWords(word_lst);
    word_dict_ = createWordDict(lexicon_);

    // Load token.txt
    fs::path token_lst = sample_data_dir / "tokens.lst";
    token_dict_ = Dictionary(token_lst);
    token_dict_.addEntry("<1>");  // for replabel emulation

    // Load lm.arpa
    fs::path lm_path = sample_data_dir / "lm.arpa";
    lm_ = std::shared_ptr<KenLM>(new KenLM(lm_path, word_dict_));
  }
  // Resouces of unittest
  LexiconMap lexicon_;
  Dictionary word_dict_;
  Dictionary token_dict_;
  std::shared_ptr<KenLM> lm_;
};

TEST_F(TestFlashlightDecoder, BasicTest) {
  ASSERT_EQ(lexicon_.size(), 26710);  // num of words + <unk>
  ASSERT_EQ(lexicon_.size(), word_dict_.entrySize());
  ASSERT_EQ(29, token_dict_.entrySize());  // 28 + 1(<1> replabel)
  ASSERT_EQ(token_dict_.indexSize(), 29);
}

TEST_F(TestFlashlightDecoder, KenLMScoreTest) {
  std::vector<std::string> sentence = {"the", "cat", "sat", "on", "the", "mat"};
  // lm_->start(0): BeginSentenceWrite(), "<s>" sos tag will be taken into
  // consideration when score.
  auto in_state = lm_->start(0);
  float total_score = 0, lm_score = 0;

  // Have priority the score with existed N-gram,
  // e.g. "<s> the" exist in arpa, then score -1.05971 directly with LogProb.
  // if N-gram is not in arpa, the score with (N-1)-gram backoff score,
  // e.g. "sat on" is not in arpa, the score "sat on" with BackoffProb("sat") +
  // LogProb ("on"), which is -0.04239096 + -2.724869 = -2.76726,
  // lm_score_tgts[3]
  std::vector<float> lm_score_tgts{-1.05971, -4.19448, -3.33383,
                                   -2.76726, -1.16237, -4.64589};
  for (int i = 0; i < sentence.size(); i++) {
    const auto& word = sentence[i];
    std::tie(in_state, lm_score) =
        lm_->score(in_state, word_dict_.getIndex(word));
    ASSERT_NEAR(lm_score, lm_score_tgts[i], 1e-5);
    total_score += lm_score;
  }
  std::tie(in_state, lm_score) = lm_->finish(in_state);  // Get </s> LogProb
  total_score += lm_score;
  ASSERT_NEAR(total_score, -19.5123, 1e-5);

  float fin_score = 0;
  sentence = {"the"};
  in_state =
      lm_->start(1);  // Get rid of <s> score when encounter start of sentence.
  std::tie(in_state, lm_score) =
      lm_->score(in_state, word_dict_.getIndex(sentence[0]));
  std::tie(in_state, fin_score) = lm_->finish(in_state);
  ASSERT_NEAR(-2.348754, fin_score, 1e-5);
  ASSERT_NEAR(-2.505692, lm_score, 1e-5);
}

TEST_F(TestFlashlightDecoder, TrieTest) {
  // Unittest of Trie
  int sil_idx = token_dict_.getIndex(kSepToken);
  auto trie = std::make_shared<Trie>(token_dict_.indexSize(), sil_idx);
  auto start_state = lm_->start(0);  // Take <s> into consideration.

  // Build LexiconTrie
  for (auto lex_entry : lexicon_) {
    auto word = lex_entry.first;
    auto word_idx = word_dict_.getIndex(word);
    auto spellings = lex_entry.second;
    float lm_score = 0.0;
    LMStatePtr dummy_state;
    std::tie(dummy_state, lm_score) = lm_->score(start_state, word_idx);
    for (auto spell : spellings) {
      auto spell_idx = tkn2Idx(spell, token_dict_, 1);  // pack "call" -> cal<1>
      trie->insert(spell_idx, word_idx, lm_score);
    }
  }

  // Search "training's"
  std::vector<int> tokens = {21, 19, 2, 10, 15, 10, 15, 8, 1, 20, 0};
  auto trie_node = trie->search(tokens);
  auto search_res = word_dict_.getEntry(trie_node->labels[0]);
  ASSERT_STREQ(search_res.c_str(), "training's");
  // Since "training's" not exists in lm, thus, the score should be
  // LogProb("<s>") + LogProb("<unk>")
  ASSERT_NEAR(trie_node->scores[0], -3.0896792, 1e-5);

  // Search "organizations"
  tokens = {16, 19, 8, 2, 15, 10, 27, 2, 21, 10, 16, 15, 20, 0};
  trie_node = trie->search(tokens);
  search_res = word_dict_.getEntry(trie_node->labels[0]);
  ASSERT_STREQ(search_res.c_str(), "organizations");
  // LogProb ("<s>") + LogProb("organizations")
  ASSERT_NEAR(trie_node->scores[0], -5.1494582, 1e-5);

  // Search "cat"
  // In this case, it would encounter more than one labels at a TrieNode, for,
  // entry "cat c a t |" and "c._a._t. cat |" shared same spellings for
  // different words. This usually won't happen when you have texts and tokens
  // both normalized before training, intuitivly.
  tokens = {4, 2, 21, 0};
  trie_node = trie->search(tokens);
  search_res = word_dict_.getEntry(trie_node->labels[0]);
  ASSERT_STREQ(search_res.c_str(), "cat");
  ASSERT_STREQ(word_dict_.getEntry(trie_node->labels[1]).c_str(), "c._a._t.");
  // LogProb("<s>") + LogProb("cat")
  ASSERT_NEAR(trie_node->scores[0], -4.5316362, 1e-5);
  // LogProb("<s>") + LogProb("<unk>"), Since c._a._t. is not in lm
  ASSERT_NEAR(trie_node->scores[1], -3.0896792, 1e-5);
  ASSERT_NEAR(trie_node->maxScore, 0.0,
              1e-5);  // MaxScore remain 0.0 since it will only be initialized
                      // value when smear.

  trie->smear(SmearingMode::MAX);
  // Smear will first assign MaxScore with LogAdd current Node scores, then if
  // SmearMode:: MAX selected, Update MaxScore with maximum MaxScore of all
  // children nodes.
  ASSERT_NEAR(trie_node->maxScore, -2.87742, 1e-5);
}

TEST_F(TestFlashlightDecoder, LexiconDecoderTest) {
  /* Build Resource of decoding */
  int unk_idx = word_dict_.getIndex(kUnkToken);  // <unk> index in word_dict
  int sil_idx = token_dict_.getIndex(kSepToken);
  token_dict_.addEntry("<blank_id>");  // Add <blank_id> for CTC decode
  int blank_idx = token_dict_.getIndex("<blank_id>");
  auto trie = std::make_shared<Trie>(token_dict_.indexSize(), sil_idx);
  auto start_state = lm_->start(false);  // "<s>" score included.

  for (auto lex_entry : lexicon_) {
    auto word = lex_entry.first;
    auto word_idx = word_dict_.getIndex(word);
    auto spellings = lex_entry.second;
    float lm_score = 0.0;
    LMStatePtr dummy_state;
    std::tie(dummy_state, lm_score) = lm_->score(start_state, word_idx);
    for (auto spell : spellings) {
      auto spell_idx = tkn2Idx(spell, token_dict_, 1);  // pack "call" -> cal<1>
      trie->insert(spell_idx, word_idx, lm_score);
    }
  }
  trie->smear(SmearingMode::MAX);

  /*------ Build Decoder ------*/
  LexiconDecoderOptions decoderOpt = {
      .beamSize = 50,                                       // beamsize
      .beamSizeToken = 50,                                  // beamsizetoken
      .beamThreshold = 100.0,                               // beamThreshold
      .lmWeight = 2.0,                                      // lmweight
      .wordScore = 2.0,                                     // lexiconscore
      .unkScore = -std::numeric_limits<float>::infinity(),  // unkscore
      .silScore = -1,                                       // silscore
      .logAdd = false,                                      // logadd
      .criterionType = CriterionType::CTC};

  std::vector<float> dummy_trans;
  auto decoder = std::make_shared<LexiconDecoder>(
      decoderOpt,   // LexiconDecoderOptions
      trie,         // Pointer of LexiconTrie
      lm_,          // Pointer of LM
      sil_idx,      // silence_index, or seperate token index, from TokenDict
      blank_idx,    // blank_id index from TokenDict
      unk_idx,      // <unk> unk_id from WordDict
      dummy_trans,  // This is required for ASG, should be dummy for CTC.
      false);       // if LAS model applied, it should be true, you got it.

  /* Initialze Dummy logits */
  int T = 235, N = 29;
  torch::Tensor logit_Tensor = torch::rand({1, T, N}).squeeze(0).contiguous();
  std::vector<float> logits_vec(
      logit_Tensor.data_ptr<float>(),
      logit_Tensor.data_ptr<float>() + logit_Tensor.numel());
  ASSERT_EQ(T * N, logits_vec.size());  // Flatten 2D logits into 1D

  /* Decode streaming-free */
  auto decoded_res = decoder->decode(&logits_vec[0], T, N);

  /* Decode streaming */
  decoder->decodeBegin();
  DecodeResult partial_res;
  for (int i = 0; i < T; i += 20) {
    decoder->decodeStep(&logits_vec[i * 20], 20, N);
    partial_res = decoder->getBestHypothesis();
    decoder->prune();  // According to docs from decoder.h
  }
  decoder->decodeEnd();
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}