# speech to text

`speech2text` is a toolkit provides multiple speech understanding system related with __automatic speech recognition__. 
Carefully designed decoupled inferface of different modules and related unittests provide scalability and robustess of customization and expansion on new systems in the future. This tookit provides model production on different tasks as below. ***Feel free to reach me 609946862@qq.com or issue if any problem encountered!***

- `CTC`: __CTC__ based automatic speech recognition system.
- `Rnnt`: __Vanila Rnn-Transduce__ system on ASR task.
- `CTC_Hybrid_Rnnt`: __CTC + Rnnt hybrid__ ASR system.
- `Pruned_Rnnt`: __Pruned Rnnt__ system proposed by `k2`, Especially, `zipformer-stateless-pruned-rnnt` achieves as same SOTA results as `k2/icefall` does on librispeech dataset.
- `SSL`: __Self-supervised learning__ pretrain framework based on [__BERT-based Speech pre-Training
with Random-projection Quantizer (BEST-RQ)__](https://arxiv.org/pdf/2202.01855) proposed by Google and leveraged on [__Universal Speech Model__](https://arxiv.org/abs/2303.01037) system, which exhibits SOTA results on speech recognition tasks across many languages.
- `CIF`: [__Continuous Integrate-Fire system__](https://arxiv.org/pdf/1905.11235) end-to-end speech recognition system leverages non-autoregressive attention-based encoder decoder providing much better decoding proformance than vanilla __AED__ system. 
- `NNLM`: __Rnn LM__ task, provide extra language infomations to decoding system during decoding stage.


## Environment set up
Build training runtime with `Docker` as below. If intend to use `conda` to build your env, run `pip install -r requirements.txt`, but this is __NOT RECOMENDED__. 
```bash
# Build docker image
docker build -t training_env:version . -f Dockerfile.build
# Start your container with docker 
docker run -itd \
    --gpus=all \
    --ipc=host \
    --name=training-runtime \
    -v /mnt:/mnt \
    training_env:0.1 /bin/bash
```

## Build systems

This toolkit provided multi different demo system config of training and inference, please check details of different tasks setting from `config/*`. `build_task.py` and `inference.py` served as entry-points to build training and inference tasks respectivly. 
- __training:__ 
```bash
CUDA_VISIBLE_DEVICES=0,1 python build_task.py \
    --training_config=config/training/zipformer_stateless_pruned_rnnt.yaml
```
- __inference:__
```bash
CUDA_VISIBLE_DEVICES=0,1 python inference.py \
    --inference_config=config/inference/zipformer_stateless_pruned_rnnt_beam_search.yaml
```
