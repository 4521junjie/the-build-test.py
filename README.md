# the-build-test.py
## ❤1、验证集和测试集的相同点和目的(The similarities and purposes of validation set and test set.)
### 相同点
1、都是由原始数据集划分出来的； 

2、都用于模型的评估和选择；

3、都用于衡量模型的泛化能力。

### 目的
验证集的主要目的是用于在训练过程中作为模型的性能评估标准，以帮助选择最佳模型。在训练过程中，模型不断地在验证集上测试，并调整参数，以达到最佳的验证准确率。验证集通常会在训练集上训练模型，并在验证集上进行模型评估，以确定哪个模型的验证集效果最好。

测试集的目的是在训练和调整模型后对模型进行最终的评估，以估计模型在实际应用中的性能表现和评估其泛化能力。测试集不会对模型的设计和参数进行调整，因此可以更好地表现模型在未见过的数据上的实际表现能力。

针对resnet34的测试脚本test.py，可以按照通常的操作流程使用训练集、验证集和测试集对模型进行评估。在训练时使用训练集和验证集，对模型进行交叉验证，确定最佳的超参数，最后使用测试集对模型的性能进行最终的评估。
``` python 
import argparse
import time
import json
import os
from argparse import Namespace

from tqdm import tqdm
from models import *
# from efficientnet_pytorch import EfficientNet
from torch import nn
from torch import optim
# from torch.optim.lr_scheduler import *
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from tools import warmup_lr


# 初始化参数
def get_args():
    """在下面初始化你的参数.
    """
    parser = argparse.ArgumentParser(description='基于Pytorch实现的分类任务')

    # exp
    parser.add_argument('--time_exp_start', type=str,
                        default=time.strftime('%m-%d-%H-%M', time.localtime(time.time())))
    parser.add_argument('--train_dir', type=str, default='data/train')
    parser.add_argument('--val_dir', type=str, default='data/val')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--save_station', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--is_mps', type=bool, default=False)
    parser.add_argument('--is_cuda', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)

    # dataset
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--data_mean', type=tuple, default=[.5, .5, .5])
    parser.add_argument('--data_std', type=tuple, default=[.5, .5, .5])

    # model
    parser.add_argument('--model', type=str, default='ResNet18',
                        choices=[
                            'ResNet18',
                            'ResNet34',
                            'ResNet50',
                            'ResNet18RandomEncoder',
                        ])

    # scheduler
    parser.add_argument('--warmup_epoch', type=int, default=1)

    # 通过json记录参数配置
    args = parser.parse_args()
    args.directory = 'dictionary/%s/Hi%s/' % (args.model, args.time_exp_start)
    log_file = os.path.join(args.directory, 'log.json')
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
    with open(log_file, 'w') as log:
        json.dump(vars(args), log, indent=4)

    # 返回参数集
    return args


class Worker:
    def __init__(self, args):
        self.opt = args

        # 判定设备
        self.device = torch.device('cuda:0' if args.is_cuda else 'cpu')
        # 是一个用来指定训练设备的参数,它可以指定你的训练是在CPU上还是GPU上进行,如果args.is_cuda为True，则将设备设置为cuda:0，否则设置为cpu
        kwargs = {
            'num_workers': args.num_workers,
            # 是一个用来指定数据加载器的参数，它可以指定你的数据加载器使用多少个线程来加载数据，这样可以提高数据加载的效率。
            # 表示设置训练过程中的数据加载器使用的进程数量。args.num_workers 是从命令行参数中获取的。它的值决定了数据加载器在读取数据时使用的并行进程数量。
            # 进程数量过高可能会导致系统资源占用过多，而影响其他进程的运行。
            'pin_memory': True,
            # 是一个用来指定数据加载器的参数，它可以指定你的数据加载器是否将数据加载到内存中
        } if args.is_cuda else {}

        # 载入数据
        train_dataset = datasets.ImageFolder(
            args.train_dir,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(256),     
                transforms.ToTensor()             
            ])
        )
        val_dataset = datasets.ImageFolder(
            args.val_dir,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(256),
                transforms.ToTensor()
                # transforms.Normalize(opt.data_mean, opt.data_std)
            ])
        )
        self.train_loader = DataLoader(
            dataset=train_dataset,
           
            batch_size=args.batch_size,
           
            shuffle=True,
                   **kwargs
        )
        self.val_loader = DataLoader(
          
            dataset=val_dataset,
           
            batch_size=args.test_batch_size,
          
            shuffle=False,
         
            **kwargs
        )

        # 挑选神经网络、参数初始化
        net = None
        if args.model == 'ResNet18':
            net = ResNet18(num_cls=args.num_classes)
        elif args.model == 'ResNet34':
            net = ResNet34(num_cls=args.num_classes)
        elif args.model == 'ResNet50':
            net = ResNet50(num_cls=args.num_classes)
        elif args.model == 'ResNet18RandomEncoder':
            net = ResNet18RandomEncoder(num_cls=args.num_classes)
        assert net is not None

        self.model = net.to(self.device)
 
        # 优化器
        self.optimizer = optim.AdamW(
           
            self.model.parameters(),
    
            lr=args.lr
          
        )

        # 损失函数
        self.loss_function = nn.CrossEntropyLoss()
  

        # warm up 学习率调整部分
        self.per_epoch_size = len(train_dataset) // args.batch_size
        self.warmup_step = args.warmup_epoch * self.per_epoch_size
        self.max_iter = args.epochs * self.per_epoch_size
        self.global_step = 0

    def train(self, epoch):
        self.model.train()
        bar = tqdm(enumerate(self.train_loader))
        for batch_idx, (data, target) in bar:
            self.global_step += 1
            data, target = data.to(self.device), target.to(self.device)

            # 训练中...
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_function(output, target)
            loss.backward()
            self.optimizer.step()
            lr = warmup_lr.adjust_learning_rate_cosine(
                self.optimizer, global_step=self.global_step,
                learning_rate_base=self.opt.lr,
                total_steps=self.max_iter,
                warmup_steps=self.warmup_step
            )

            # 更新进度条
            bar.set_description(
                'train epoch {} >> [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tlr: {:.6f} >> '.format(
                    epoch,
                    batch_idx * len(data),
                    len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item(),
                    lr
                )
            )
        bar.close()

    def val(self):
        self.model.eval()
        validating_loss = 0
        num_correct = 0
        with torch.no_grad():
            bar = tqdm(self.val_loader)
            for data, target in bar:
                # 测试中...
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                validating_loss += self.loss_function(output, target).item()  # 累加 batch loss
                pred = output.argmax(dim=1, keepdim=True)  # 获取最大概率神经元下标
                num_correct += pred.eq(target.view_as(pred)).sum().item()
            bar.close()

        # 打印验证结果
        validating_loss /= len(self.val_loader)
        print('val >> Average loss: {:.4f}, Accuracy: {}/{} ({:.03f}%)\n'.format(
            validating_loss,
            num_correct,
            len(self.val_loader.dataset),
            100. * num_correct / len(self.val_loader.dataset))
        )

        # 返回重要信息，用于生成模型保存命名
        return 100. * num_correct / len(self.val_loader.dataset), validating_loss


if __name__ == '__main__':
    # 初始化
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(0)
    args = get_args()
    worker = Worker(args=args)
   

    # 训练与验证
    for epoch in range(1, args.epochs + 1):
        # 中的 epoch 是一种用于编程的循环命令。它用于重复一组指令一定次数。例如，如果要运行程序 10 次，则可以使用此命令执行此操作。它是自动化任务并确保程序多次正确运行的有用工具。
        worker.train(epoch)
        val_acc, val_loss = worker.val()
        # 是一种机器学习技术，用于测量模型在验证集上的性能。
        # val_acc是模型在验证集上的准确率，val_loss是模型在验证集上的损失值。这些指标可以用来衡量模型的性能，以便确定模型是否可以在实际应用中使用。
        if epoch > args.save_station:
            save_dir = args.directory + '%s-epochs-%d-model-val-acc-%.3f-loss-%.6f.pt' \
                       % (args.model, epoch, val_acc, val_loss)
            torch.save(worker.model, save_dir)
```
![image](https://github.com/4521junjie/the-build-test.py/assets/119326710/f145cba5-fa7f-47c9-8b06-0b7d3d46bec6)
