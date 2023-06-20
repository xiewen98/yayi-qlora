# QLoRA_YaYi

**介绍**

本项目是基于闻歌开源大模型YaYi做QLoRA微调。

QLORA先将base_model做NF4（NormalFloat-4Bit）量化，再进行LoRA微调。因此，需要安装最新的bnb与transform库。

NF4与FP4类似，是一种4-bit量化。相比FP4，NF4量化的分布更适应于正态分布的数据。

![1](https://github.com/xiewen98/QLoRA_YaYi/tree/main/img/1.png)

经过NF4与FP4量化后的模型可以观察到，在bnb库中是将两个4bit量化的值合为一个uint8类型，因此在量化过程中会发现模型中QKV权重矩阵的size从4096* 12288变成25165824* 1，刚好size少了一倍，并且量化的值也不是0-15而是0-255。

如00010001 uint8值为17，但是在源码中是先进行右移四位，因此也就是将uint8变成2个uint4进行解码。

![2](https://github.com/xiewen98/QLoRA_YaYi/tree/main/img/2.png)

![3](https://github.com/xiewen98/QLoRA_YaYi/tree/main/img/3.png)

**运行方式**

1、下载github链接至本地

```
git clone https://github.com/xiewen98/QLoRA_YaYi.git
cd QLoRA_YaYi
```

2、安装环境

```
pip install -r requirements.txt
```

3、LoRA微调

数据集为参考chatglm-6b ptuning微调所提供的广告数据集

```
CUDA_VISIBLE_DEVICES=0 python train.py
```

相关超参说明

```
--model_name_or_path,  help='model path'
--prompt_column,  help='get the column names for input/target.'
--response_column, help='answer in the data')
--history_column,  help='history in the data'
--ignore_pad_token_for_loss, help='add pad_token need to calculate loss'
--max_target_length,  help='output maximum length'
--max_source_length, help='input maximum length'
--train_file, help='train dataset path'
--quan, help='nf4 or fp4'
--double_quant, help='double quant'
--lora_r, help='low rank'
--lora_alpha, help='normalized lora rank '
--epoch, help='train epoch'
--batch_size, help='batch size'
--gradient_steps, help='end_batch is batch_size*gradient_steps'
--out_dir, help='lora save path'
--lr, help='learn rate'
```

```
trainable params: 7864320 || all params: 4056178688 || trainable%: 0.19388494947883322
```

在lora_r设置为16时,仅需训练0.2%的参数。注意：训练参数的增多不太影响GPU显存占用。

训练之后，在best_model下会输出lora模型与lora模型的配置文件。以lora_r=16为例，lora权重大小仅为30M。

4、验证

```
CUDA_VISIBLE_DEVICES=0 python eval.py
```

```
验证部分，采用指标
rouge-1、rouge-2、rouge-l、bleu-4"
```

5、推理

```
CUDA_VISIBLE_DEVICES=0 python predict.py
```

**所需资源**

​	YaYi验证与推理：20G显存

​	YaYi+QLORA微调：13G显存
​	YaYi+QLORA验证与推理：9G显存



**后续工作**

LLM在终身学习中缓解出现的灾难遗忘现象。（包括NLP中各领域任务以及多模态任务）



## 致谢

非常感谢以下作者的开源与资料分享

https://github.com/wenge-research/YaYi

https://github.com/artidoro/qlora

https://readpaper.feishu.cn/docx/CrMGdSVPKow5d1x1XQMcJioRnQe

https://github.com/THUDM/ChatGLM-6B

