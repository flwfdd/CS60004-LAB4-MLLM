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

用类似的方法构造 hard val 322 条。


## Task 2

bs 16 256张图片每张图片 8 条样本
理论上2048条筛选后A1789条B1522条

训练参数为bs 16 lr 5e-6 512 samples

A：
Answer Accuracy (val):                                                                             
  total   = 0.6970 (n=1000)
  bbox    = 0.6894 (n=937)
  no_bbox = 0.8095 (n=63)
Trainable: 2,205,754,368 / 2,205,754,368 (100.0000%)
Peak GPU memory: 21.66 GiB
Train wall time: 292.6 s

B：
Answer Accuracy (val):                                                                             
  total   = 0.7020 (n=1000)
  bbox    = 0.6937 (n=937)
  no_bbox = 0.8254 (n=63)
Trainable: 2,205,754,368 / 2,205,754,368 (100.0000%)
Peak GPU memory: 21.66 GiB
Train wall time: 297.9 s


AB:
Answer Accuracy (val):                                                                             
  total   = 0.6920 (n=1000)
  bbox    = 0.6841 (n=937)
  no_bbox = 0.8095 (n=63)
Trainable: 2,205,754,368 / 2,205,754,368 (100.0000%)
Peak GPU memory: 21.66 GiB
Train wall time: 297.4 s

1A:
Answer Accuracy (val):                                                                             
  total   = 0.6980 (n=1000)
  bbox    = 0.6884 (n=937)
  no_bbox = 0.8413 (n=63)
Trainable: 2,205,754,368 / 2,205,754,368 (100.0000%)
Peak GPU memory: 21.66 GiB
Train wall time: 296.7 s

1B:
Answer Accuracy (val):                                                                             
  total   = 0.7000 (n=1000)
  bbox    = 0.6905 (n=937)
  no_bbox = 0.8413 (n=63)
Trainable: 2,205,754,368 / 2,205,754,368 (100.0000%)
Peak GPU memory: 21.66 GiB
Train wall time: 300.9 s

1AB bs32 lr1e-5 1024 samples:
Answer Accuracy (val):                                                                             
  total   = 0.6900 (n=1000)
  bbox    = 0.6809 (n=937)
  no_bbox = 0.8254 (n=63)
Trainable: 2,205,754,368 / 2,205,754,368 (100.0000%)
Peak GPU memory: 21.66 GiB
Train wall time: 593.3 s

## Task 3

bbox_prompt qwen3vl:
Answer Accuracy:
  total   : 843/1000 = 0.8430
  bbox    : 792/937 = 0.8453
  no_bbox : 51/63 = 0.8095

bs 16 lr 5e-6 512 samples:
在 no_bbox 上训练然后 bbox 上推理：
Answer Accuracy:
  total   : 832/1000 = 0.8320
  bbox    : 780/937 = 0.8324
  no_bbox : 52/63 = 0.8254

在 bbox 上训练：

A_connector_only:
bs 16 lr 1e-4 512 samples:
Answer Accuracy (val):                                                                             
  total   = 0.8550 (n=1000)
  bbox    = 0.8581 (n=937)
  no_bbox = 0.8095 (n=63)
Trainable: 12,595,200 / 2,205,754,368 (0.5710%)
Peak GPU memory: 19.17 GiB
Train wall time: 273.6 s

B_connector_language:
bs 16 lr 5e-6 512 samples:
Answer Accuracy (val):                                                                             
  total   = 0.8690 (n=1000)
  bbox    = 0.8719 (n=937)
  no_bbox = 0.8254 (n=63)
Trainable: 1,901,742,080 / 2,205,754,368 (86.2173%)
Peak GPU memory: 19.36 GiB
Train wall time: 376.3 s

C_vision_connector:
bs 16 lr 1e-5 512 samples:
Answer Accuracy (val):                                                                             
  total   = 0.8530 (n=1000)
  bbox    = 0.8538 (n=937)
  no_bbox = 0.8413 (n=63)
Trainable: 316,607,488 / 2,205,754,368 (14.3537%)
Peak GPU memory: 21.20 GiB
Train wall time: 498.5 s

D_full:
bs 16 lr 5e-6 512 samples:
Answer Accuracy (val):                                                                             
  total   = 0.8720 (n=1000)
  bbox    = 0.8751 (n=937)
  no_bbox = 0.8254 (n=63)
Trainable: 2,205,754,368 / 2,205,754,368 (100.0000%)
Peak GPU memory: 21.63 GiB
Train wall time: 603.2 s

val中bbox数量分布：
0 bbox:  63  (6.3%)
1 bbox: 394  (39.4%)
2 bbox: 451  (45.1%)
3 bbox:  91  (9.1%)
4 bbox:   1  (0.1%)

以下均在 task1/D_full 基础上训练

仅bbox
[pre-train @ optim_step=0] total=0.8320 (n=1000), bbox=0.8324 (n=937), no_bbox=0.8254 (n=63)       
train[D_full]: 100%|██████████████████████| 512/512 [14:57<00:00,  1.75s/it, loss=0.0003, optim=31]
Answer Accuracy (val):                                                                             
  total   = 0.8670 (n=1000)
  bbox    = 0.8687 (n=937)
  no_bbox = 0.8413 (n=63)
Trainable: 2,205,754,368 / 2,205,754,368 (100.0000%)
Peak GPU memory: 21.64 GiB
Train wall time: 601.8 s

same_box:
uv run task_scripts/task1.py --task3-experiment same_box --freeze-config D_full --max-train-samples 512 --batch-size 16 --lr 5e-6
[pre-train @ optim_step=0] total=0.6880 (n=1000), bbox=0.6788 (n=937), no_bbox=0.8254 (n=63)       
train[D_full]: 100%|██████████████████████| 512/512 [14:31<00:00,  1.70s/it, loss=0.0012, optim=31]
Answer Accuracy (val):                                                                             
  total   = 0.6810 (n=1000)
  bbox    = 0.6702 (n=937)
  no_bbox = 0.8413 (n=63)
Trainable: 2,205,754,368 / 2,205,754,368 (100.0000%)
Peak GPU memory: 21.60 GiB
Train wall time: 587.0 s

color_box:
uv run task_scripts/task1.py --task3-experiment color_box --freeze-config D_full --max-train-samples 512 --batch-size 16 --lr 5e-6
[pre-train @ optim_step=0] total=0.6860 (n=1000), bbox=0.6766 (n=937), no_bbox=0.8254 (n=63)       
train[D_full]: 100%|██████████████████████| 512/512 [14:32<00:00,  1.70s/it, loss=0.0009, optim=31]
Answer Accuracy (val):                                                                             
  total   = 0.6860 (n=1000)
  bbox    = 0.6756 (n=937)
  no_bbox = 0.8413 (n=63)
Trainable: 2,205,754,368 / 2,205,754,368 (100.0000%)
Peak GPU memory: 21.60 GiB
Train wall time: 587.4 s

color_box_label:
uv run task_scripts/task1.py --task3-experiment color_box_label --freeze-config D_full --max-train-samples 512 --batch-size 16 --lr 5e-6
[pre-train @ optim_step=0] total=0.7930 (n=1000), bbox=0.7908 (n=937), no_bbox=0.8254 (n=63)       
train[D_full]: 100%|██████████████████████| 512/512 [14:32<00:00,  1.70s/it, loss=0.0004, optim=31]
Answer Accuracy (val):                                                                             
  total   = 0.8220 (n=1000)
  bbox    = 0.8218 (n=937)
  no_bbox = 0.8254 (n=63)
Trainable: 2,205,754,368 / 2,205,754,368 (100.0000%)
Peak GPU memory: 21.60 GiB
Train wall time: 587.8 s

same_box_bbox_prompt:
uv run task_scripts/task1.py --task3-experiment same_box_bbox_prompt --freeze-config D_full --max-train-samples 512 --batch-size 16 --lr 5e-6
[pre-train @ optim_step=0] total=0.8050 (n=1000), bbox=0.8036 (n=937), no_bbox=0.8254 (n=63)       
train[D_full]: 100%|██████████████████████| 512/512 [15:00<00:00,  1.76s/it, loss=0.0004, optim=31]
Answer Accuracy (val):                                                                             
  total   = 0.8600 (n=1000)
  bbox    = 0.8613 (n=937)
  no_bbox = 0.8413 (n=63)
Trainable: 2,205,754,368 / 2,205,754,368 (100.0000%)
Peak GPU memory: 21.64 GiB
Train wall time: 603.5 s

color_box_color_prompt:
uv run task_scripts/task1.py --task3-experiment color_box_color_prompt --freeze-config D_full --max-train-samples 512 --batch-size 16 --lr 5e-6
[pre-train @ optim_step=0] total=0.7780 (n=1000), bbox=0.7748 (n=937), no_bbox=0.8254 (n=63)       
train[D_full]: 100%|██████████████████████| 512/512 [15:02<00:00,  1.76s/it, loss=0.0005, optim=31]
Answer Accuracy (val):                                                                             
  total   = 0.8520 (n=1000)
  bbox    = 0.8527 (n=937)
  no_bbox = 0.8413 (n=63)
Trainable: 2,205,754,368 / 2,205,754,368 (100.0000%)
Peak GPU memory: 21.64 GiB
Train wall time: 604.4 s

color_box_color_prompt_without_bbox:
[pre-train @ optim_step=0] total=0.7950 (n=1000), bbox=0.7930 (n=937), no_bbox=0.8254 (n=63)       
train[D_full]: 100%|██████████████████████| 512/512 [14:37<00:00,  1.71s/it, loss=0.0004, optim=31]
Answer Accuracy (val):                                                                             
  total   = 0.8600 (n=1000)
  bbox    = 0.8613 (n=937)
  no_bbox = 0.8413 (n=63)
Trainable: 2,205,754,368 / 2,205,754,368 (100.0000%)
Peak GPU memory: 21.62 GiB
Train wall time: 589.9 s


bs 64 lr 5e-6 4096 samples:
[pre-train @ optim_step=0] total=0.8700 (n=100), bbox=0.8723 (n=94), no_bbox=0.8333 (n=6)
train[D_full]: 100%|██████████████████| 4096/4096 [1:20:54<00:00,  1.19s/it, loss=0.0002, optim=63]
Answer Accuracy (val):
  total   = 0.8740 (n=1000)
  bbox    = 0.8762 (n=937)
  no_bbox = 0.8413 (n=63)
Trainable: 2,205,754,368 / 2,205,754,368 (100.0000%)
Peak GPU memory: 21.64 GiB
Train wall time: 4824.4 s

先用 task1/D_full 以 bbox_prompt 模式评估所有 task3 训练数据，从 8000 条中筛选出 1319 条 hard cases。

bs 64 lr 5e-6 训 hard cases，val一直在掉。
换成 1e-6 512 samples
[pre-train @ optim_step=0] total=0.8700 (n=100), bbox=0.8723 (n=94), no_bbox=0.8333 (n=6)          
[periodic @ optim_step=10] total=0.8700 (n=100), bbox=0.8723 (n=94), no_bbox=0.8333 (n=6)          
[periodic @ optim_step=20] total=0.8700 (n=100), bbox=0.8723 (n=94), no_bbox=0.8333 (n=6)          
[periodic @ optim_step=30] total=0.8800 (n=100), bbox=0.8830 (n=94), no_bbox=0.8333 (n=6)          
train[D_full]: 100%|██████████████████████| 512/512 [11:47<00:00,  1.38s/it, loss=0.4499, optim=31]
Answer Accuracy (val):                                                                             
  total   = 0.8340 (n=1000)
  bbox    = 0.8346 (n=937)
  no_bbox = 0.8254 (n=63)
Trainable: 2,205,754,368 / 2,205,754,368 (100.0000%)
Peak GPU memory: 21.63 GiB
Train wall time: 588.7 s

参考[VoCoT](https://www.alphaxiv.org/abs/2405.16919)设计提示词测试COT模式：
Saved 100 predictions → outputs/task3/train_qwen3vl_bbox_prompt.jsonl
Answer Accuracy:
  total   : 84/100 = 0.8400
  bbox    : 79/94 = 0.8404
  no_bbox : 5/6 = 0.8333
Saved 100 predictions → outputs/task3/train_qwen3vl_cot.jsonl
Answer Accuracy:
  total   : 81/100 = 0.8100
  bbox    : 76/94 = 0.8085
  no_bbox : 5/6 = 0.8333

无法达到非COT效果

直接把 task1/D_full 交了一发。	77.5	581	排名第二。


