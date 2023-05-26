# the-build-test.py
## ❤1、验证集和测试集的相同点和目的(The similarities and purposes of validation set and test set.)
### 相同点
1、都是由原始数据集划分出来的； 

2、都用于模型的评估和选择；

3、都用于衡量模型的泛化能力。

### 目的
验证集的主要目的是用于在训练过程中作为模型的性能评估标准，以帮助选择最佳模型。在训练过程中，模型不断地在验证集上测试，并调整参数，以达到最佳的验证准确率。
验证集通常会在训练集上训练模型，并在验证集上进行模型评估，以确定哪个模型的验证集效果最好。

测试集的目的是在训练和调整模型后对模型进行最终的评估，以估计模型在实际应用中的性能表现和评估其泛化能力。
测试集不会对模型的设计和参数进行调整，因此可以更好地表现模型在未见过的数据上的实际表现能力。
### 总结
针对resnet34的测试脚本test.py，可以按照通常的操作流程使用训练集、验证集和测试集对模型进行评估。在训练时使用训练集和验证集，对模型进行交叉验证，确定最佳的超参数，最后使用测试集对模型的性能进行最终的评估。
### train0代码(详细代码在仓库中)
``` python 
import argparse
...
from tqdm import tqdm
...
from tools import warmup_lr
# 初始化参数
def get_args():
    """在下面初始化你的参数.
    """
    parser = argparse.ArgumentParser(description='基于Pytorch实现的分类任务')

    # exp
  ...

    # dataset
  ...

    # model
  ...

    # 通过json记录参数配置
  ...

    # 返回参数集
    return args


class Worker:
    def __init__(self, args):
        self.opt = args

        # 判定设备
      ...
        # 载入数据
       ...
        # 挑选神经网络、参数初始化
      ...
 
        # 优化器
     ...
        # 损失函数
        self.loss_function = nn.CrossEntropyLoss()
  

        # warm up 学习率调整部分
    ...

            # 训练中...
           ...
            # 更新进度条
          ...
        # 打印验证结果
     ...

        # 返回重要信息，用于生成模型保存命名
      ...
    # 初始化
  ...
   

    # 训练与验证
   ...
```
![image](https://github.com/4521junjie/the-build-test.py/assets/119326710/f145cba5-fa7f-47c9-8b06-0b7d3d46bec6)
### 将train0.py更改后得到train1.py（展示修改部分）

```python
...
from models.ResNet34_update import *
...
# from torch.optim.lr_scheduler import *
...


# 初始化参数
def get_args():
    """在下面初始化你的参数.
    """
    parser = argparse.ArgumentParser(description='基于Pytorch实现的分类任务')

    # exp
...

    # dataset
 ...
    # model
    parser.add_argument('--model', type=str, default='ResNet34')

    # scheduler
...

    # 通过json记录参数配置
  ...

    # 返回参数集
    return args


class Worker:
    def __init__(self, args):
        self.opt = args

        # 判定设备
     ...
        # 载入数据
      ...

        # 挑选神经网络、参数初始化
        net = ResNet34()
    ...

        # 优化器
  ...

        # 损失函数
        self.loss_function = nn.CrossEntropyLoss()

        # warm up 学习率调整部分
...

            # 训练中...
...

            # 更新进度条
...

        # 打印验证结果
...

        # 返回重要信息，用于生成模型保存命名
...

    # 训练与验证
...
      ```
### 运行train_update.py后得到的结果（ResNet34模型）
![image](https://github.com/4521junjie/the-build-test.py/assets/119326710/16a17fcb-3076-44bd-8f05-0bb8bb476a82)
![image](https://github.com/4521junjie/the-build-test.py/assets/119326710/e7a7abce-c514-4676-b0fa-3c5d7a2c2823)





