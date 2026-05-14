## Task 0

```
--- Configuration: A_connector_only ---
Total params: 2,205,754,368
Trainable params: 12,595,200
Trainable ratio: 0.5710%
----------------------------------------
--- Configuration: B_connector_language ---
Total params: 2,205,754,368
Trainable params: 1,901,742,080
Trainable ratio: 86.2173%
----------------------------------------
--- Configuration: C_vision_connector ---
Total params: 2,205,754,368
Trainable params: 316,607,488
Trainable ratio: 14.3537%
----------------------------------------
--- Configuration: D_full ---
Total params: 2,205,754,368
Trainable params: 2,205,754,368
Trainable ratio: 100.0000%
----------------------------------------
```

## Task 1

2. data/models/InternVL2-2B/modeling_internlm2.py
4.50 以后 PreTrainedModel 不再带 GenerationMixin，InternLM2ForCausalLM 上也就没有 generate。
与当前 LlamaForCausalLM(…, GenerationMixin) 写法对齐，改为：
from transformers.generation.utils import GenerationMixin
class InternLM2ForCausalLM(InternLM2PreTrainedModel, GenerationMixin):
这样 language_model.generate(...) 在 InternVL 里可以继续用。


val_2b_no_bbox: Answer Accuracy: 286/1000 = 0.286
val_8b_no_bbox: Answer Accuracy: 682/1000 = 0.682
val_2b_bbox_prompt: Answer Accuracy: 148/1000 = 0.148
val_8b_bbox_prompt: Answer Accuracy: 845/1000 = 0.845

查看https://github.com/OpenGVLab/InternVL中的官方脚本，基本采用了batchsize=128,lr=4e-5：

爆显存需要加gradient_checkpointing

D_full:
bs 128 lr 4e-5
Answer Accuracy (val, n=1000): 0.5880
Trainable: 2,205,754,368 / 2,205,754,368 (100.0000%)
Peak GPU memory: 21.65 GiB
Train wall time: 588.4 s

bs 64 lr 2e-5
Answer Accuracy (val):                                                                             
  total   = 0.6420 (n=1000)
  bbox    = 0.6307 (n=937)
  no_bbox = 0.8095 (n=63)
Trainable: 2,205,754,368 / 2,205,754,368 (100.0000%)
Peak GPU memory: 21.65 GiB
Train wall time: 585.3 s

bs 32 lr 2e-5
Answer Accuracy (val, n=1000): 0.6410
Trainable: 2,205,754,368 / 2,205,754,368 (100.0000%)
Peak GPU memory: 21.65 GiB
Train wall time: 590.1 s

bs 32 lr 1e-5
Answer Accuracy (val, n=1000): 0.6910
Trainable: 2,205,754,368 / 2,205,754,368 (100.0000%)
Peak GPU memory: 21.65 GiB
Train wall time: 589.5 s

bs 32 lr 5e-6
Answer Accuracy (val, n=1000): 0.6920
Trainable: 2,205,754,368 / 2,205,754,368 (100.0000%)
Peak GPU memory: 21.65 GiB
Train wall time: 589.8 s

bs 32 lr 1e-6
Answer Accuracy (val, n=1000): 0.6930
Trainable: 2,205,754,368 / 2,205,754,368 (100.0000%)
Peak GPU memory: 21.65 GiB
Train wall time: 594.3 s

bs 16 lr 5e-6
Answer Accuracy (val):                                                                             
  total   = 0.6950 (n=1000)
  bbox    = 0.6873 (n=937)
  no_bbox = 0.8095 (n=63)
Trainable: 2,205,754,368 / 2,205,754,368 (100.0000%)
Peak GPU memory: 21.65 GiB
Train wall time: 592.5 s

bs 16 lr 5e-6 512 samples
Answer Accuracy (val):                                                                             
  total   = 0.6990 (n=1000)
  bbox    = 0.6905 (n=937)
  no_bbox = 0.8254 (n=63)
Trainable: 2,205,754,368 / 2,205,754,368 (100.0000%)
Peak GPU memory: 21.65 GiB
Train wall time: 296.6 s

bs 8 lr 1e-6 512 samples
Answer Accuracy (val):                                                                             
  total   = 0.6950 (n=1000)
  bbox    = 0.6873 (n=937)
  no_bbox = 0.8095 (n=63)
Trainable: 2,205,754,368 / 2,205,754,368 (100.0000%)
Peak GPU memory: 21.65 GiB
Train wall time: 296.2 s

bs 8 lr 1e-6 256 samples
Answer Accuracy (val):                                                                             
  total   = 0.6640 (n=1000)
  bbox    = 0.6574 (n=937)
  no_bbox = 0.7619 (n=63)
Trainable: 2,205,754,368 / 2,205,754,368 (100.0000%)
Peak GPU memory: 21.65 GiB
Train wall time: 150.7 s

统一使用 bs32 lr 1e-5 1024 samples

A_connector_only:
Answer Accuracy (val, n=1000): 0.6930
Trainable: 12,595,200 / 2,205,754,368 (0.5710%)
Peak GPU memory: 18.13 GiB
Train wall time: 286.9 s


B_connector_language:
Answer Accuracy (val, n=1000): 0.6930
Trainable: 1,901,742,080 / 2,205,754,368 (86.2173%)
Peak GPU memory: 19.38 GiB
Train wall time: 384.2 s

C_vision_connector:
Answer Accuracy (val, n=1000): 0.7010
Trainable: 316,607,488 / 2,205,754,368 (14.3537%)
Peak GPU memory: 20.15 GiB
Train wall time: 487.5 s

D_full:
Answer Accuracy (val, n=1000): 0.6910
Trainable: 2,205,754,368 / 2,205,754,368 (100.0000%)
Peak GPU memory: 21.65 GiB
Train wall time: 589.5 s

![](images/task1.png)

都是很快收敛，可能都是格式带来的收益。查看未微调的也确实有：
{"id": "02853535", "prediction": "The animal on the hill is black.", "answer": "black", "correct": false, "has_bbox": true}

先通过 bs 16 lr 5e-6 512 samples 微调，然后将评测失败的样本作为 hard cases，筛选出 2238 条。

bs 16 lr 5e-6 512 samples
Answer Accuracy (val):                                                                             
  total   = 0.6990 (n=1000)
  bbox    = 0.6916 (n=937)
  no_bbox = 0.8095 (n=63)
Trainable: 2,205,754,368 / 2,205,754,368 (100.0000%)
Peak GPU memory: 21.65 GiB
Train wall time: 293.5 s

观察到 loss 一直有较大波动，最终效果并没有明显提升。