# the-build-test.py
## 1、 ❤The similarities and purposes of validation set and test set
### 1.1 similarities
1、both are subsets partitioned from the original dataset； 

2、both are used for model evaluation and selection；

3、both are used to measure the generalization ability of models.。

### 1.2 Purpose
The primary purpose of the validation dataset is to serve as a performance evaluation metric for a model during the training process, helping to select the best model. During training, the model is continuously tested on the validation dataset and parameters are adjusted to achieve the best validation accuracy.
The validation dataset is typically used to train a model on the training dataset and evaluate its performance on the validation dataset, in order to determine which model performs best on the validation dataset.

The purpose of a testing dataset is to perform the final evaluation of a model after training and tuning, in order to estimate the model’s performance and evaluate its generalization ability in real-world applications.
The testing dataset does not adjust the model’s design and parameters, thus it can better represent the true performance of the model on unseen data.

### 1.3 Summary
For the test script test.py targeted at resnet34, the model can be evaluated using the usual workflow with training, validation, and test data sets.

## 2、🧡实践部分
### 2.1train0 Code (Detailed code is in the repository)

The first step of the whole experiment is to run train0.py without encountering any issues. It has already been successfully run on the basis of the previous work。
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
### Screenshot of successful execution
![image](https://github.com/4521junjie/the-build-test.py/assets/119326710/f145cba5-fa7f-47c9-8b06-0b7d3d46bec6)


### 2.2train0.py was modified to obtain train1.py（The modified parts are shown below.）
The second step is to apply the ResNet34 model and save the trained model
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
### Screenshot of successful execution
![image](https://github.com/4521junjie/the-build-test.py/assets/119326710/5c7672fb-2baa-4521-b36e-9796a7656f21)

### 2.3The code for train1.py was modified and renamed to test.py.（The detailed code is in the repository）
The last step is also a crucial one. Initially, I couldn't understand it and it wasn't until later that I realized the meaning of the sentence "你会发现 val 和 test 的步骤在本质上是一模一样的". So, I deleted and modified some parts, and encountered some issues such as file paths and naming. After troubleshooting, I finally resolved them with the help of my classmate Li. Thank you very much!

```python
import argparse
...
# 初始化参数
def get_args():
  ...
    # model
    parser.add_argument('--model', type=str, default='ResNet34')
    # scheduler
  ...
    # 通过json记录参数配置
  ...
    # 返回参数集
    return arg
class Worker:
    def __init__(self, args):
        self.opt = args
        # 判定设备
       ... )
        # 挑选神经网络、参数初始化
     ...
        # 优化器
     ...)
        # 损失函数
        self.loss_function = nn.CrossEntropyLoss()
    def test(self):
        self.model.eval()
        validating_loss = 0
       ...
                # 测试中...
              ...
        # 打印验证结果
       ...
        # 返回重要信息，用于生成模型保存命名
      ...
if __name__ == '__main__':
    # 初始化
  ...
    # 训练与验证
    for epoch in range(1, args.epochs + 1):
        test_acc, test_loss = worker.test()
        if epoch > args.save_station:
            save_dir = args.directory + '%s-epochs-%d-model-test-acc-%.3f-loss-%.6f.pt' \
                       % (args.model, epoch, test_acc, test_loss)
            torch.save(worker.model, save_dir)
```
### Screenshot of successful execution
![image](https://github.com/4521junjie/the-build-test.py/assets/119326710/a677dfe7-424c-4261-b8ed-04be63aa3eca)
## 3、 💛测试结果
1、The model has an average loss of 0.6966 and an accuracy of 0/7 after testing on the validation set, which means the prediction accuracy of the model is 0% on the test set, indicating poor performance on the new data.

2、This means that the model may perform well on the training data, but performs poorly on new data and is unable to accurately predict the target variable. Further model optimization or adjustments are needed to improve the prediction capability of the model.

3、More training data, deeper neural network structures, better optimizers, or adjusting hyperparameters might be needed to improve the performance of the model.






























