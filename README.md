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
            # 是一个用来加载图像数据集的函数，它可以从指定的路径中加载图像数据集，从而为训练模型提供训练数据。
            # 指定了训练数据的目录，args.train_dir是一个参数，代表了训练数据的目录路径，该路径会被传递给训练代码中的相关函数，使得数据可以被正确地读取和使用
            args.train_dir,
            # 它可以帮助机器学习算法从图像中提取特征，从而更好地做出预测
            transform=transforms.Compose([
                transforms.RandomResizedCrop(256),
                # 对图片进行随机大小裁剪，并将其调整为256x256的大小。
                # 是一种图像处理技术，它可以将图像的大小调整为256像素，以便在不改变图像的像素比例的情况下改变图像的大小。
                # 它的主要用途是在计算机视觉和机器学习领域，用于将图像调整为统一的大小，以便更好地进行分析和处理。
                # 它还可以用于图像缩放，以便在网站上更好地显示图像。
                # Transform是一个Python库，它提供了一组用于转换图像的工具。
                # 在这种情况下，RandomRessizeCrop（256） 变换用于将图像随机裁剪为 256 像素大小。
                # 这对于数据增强非常有用，数据增强是从现有数据创建其他数据以增加数据集大小的过程。
                # 这可以通过提供更多数据进行训练来帮助提高机器学习模型的准确性。
                transforms.ToTensor()
                # 将PIL图像或numpy.ndarray转换为张量（Tensor）类型。transforms.ToTensor()将图像像素的值从0-255转换为0-1的范围内的浮点数，并将其存储为张量。
                # 是一种数据转换方法，它可以将数据从一种格式转换为另一种格式。
                # 它的作用是将数据从原始格式转换为张量（Tensor）格式，以便用于机器学习和深度学习模型。
                # 张量是一种多维数组，它可以更有效地表示和处理数据，这样机器学习和深度学习模型就可以更快地处理和分析数据。
                # transforms.Normalize(opt.data_mean, opt.data_std)
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
            # 将定义数据集用于模型训练。train_dataset是一个变量，它包含了我们的训练数据。在这里，我们使用dataset=train_dataset来指定我们要使用的数据集。这个操作将数据集传递给模型训练器，这样它就可以使用数据来训练我们的模型。
            batch_size=args.batch_size,
            # batch_size是指每一次模型训练时，输入的数据分成的小块的大小。这个值决定了一次训练中跑多少个样本。
            shuffle=True,
            # shuffle=True在模型训练中的作用是使每个epoch中的训练数据顺序随机化，从而增加训练的随机性和稳定性。这样可以防止模型在顺序训练过程中出现输入相关的过拟合现象。
            **kwargs
        )
        self.val_loader = DataLoader(
            # self.val_loader = DataLoader 是一个 Python 库，用于加载和组织机器学习模型的数据。它有助于快速有效地将数据加载到模型中，并使其更易于处理和分析。
            # 它可用于从各种源加载数据，例如 CSV 文件、数据库和其他源。它还有助于将数据组织成批次，从而更轻松地训练模型。此外，它还可用于执行数据增强，这有助于提高模型的准确性。
            dataset=val_dataset,
            # 数据集是以特定方式组织和格式化的数据集合。在这种情况下，val_dataset是以对特定目的有用的方式组织和格式化的数据集合。
            batch_size=args.test_batch_size,
            # 是指在训练过程的每次迭代中使用的数据点数。这很重要，因为它会影响模型的准确性以及训练模型所需的时间。
            # 批大小越大，每次迭代中使用的数据点就越多，这可以产生更准确的模型。
            # 训练模型也需要更长的时间。另一方面，较小的批量大小可以导致更快的训练，但模型的准确性可能较低。
            shuffle=False,
            # 它不会打乱数据集中的数据，而是按照原来的顺序加载数据
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
        # 将神经网络从一个设备移动到另一个设备。它用于允许神经网络在不同的设备上运行，例如计算机或移动设备，以便它可以用于不同的应用程序。
        # 这行代码对于允许神经网络在不同的上下文和不同目的中使用非常重要。
        # 将模型(net)移动到指定的设备(device)上进行训练

        # 优化器
        self.optimizer = optim.AdamW(
            # AdamW是一种优化器，它是Adam优化器的变体，用于深度学习模型的训练。
            # 它的主要优点是它可以更有效地调整学习率，从而更快地收敛到最优解。AdamW优化器可以帮助模型更快地收敛，从而提高模型的准确性和性能。
            self.model.parameters(),
            # 它可以帮助我们识别模式，并且可以提供更准确的预测结果。它可以帮助我们更好地理解数据，并且可以提供更准确的预测结果。
            # 它还可以帮助我们发现潜在的规律，从而改善我们的决策过程。
            lr=args.lr
            # 是一个参数，它可以用来控制机器学习算法的学习率，也就是说它可以控制算法的收敛速度。
            # 它可以帮助算法更快地收敛，从而提高算法的准确性和性能。
        )

        # 损失函数
        self.loss_function = nn.CrossEntropyLoss()
        # 定义模型训练过程中的损失函数，即交叉熵损失函数。它的作用是计算模型预测结果与目标值之间的差异，并根据这个差异来反向传播更新模型参数。
        # 交叉熵损失函数适用于多分类任务，因为它能够将模型预测的概率分布与真实概率分布之间的差异最小化。

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
    # 在模型训练中，worker = Worker(args=args)的作用是创建一个Worker对象，以便在本地或远程计算机上进行多进程训练。
    # Worker对象是 PyTorch 的 DistributedDataParallel 组件的一部分，它利用多进程并行计算来加速模型训练过程，并帮助在多个GPU、多个计算机上训练模型
    # 是一种用于创建“worker”对象的编程代码。此对象用于执行特定任务，例如运行程序或执行特定命令。args 参数用于传入工作人员完成任务所需的任何其他信息。
    # 这种类型的代码通常用于分布式计算，其中任务在多台计算机之间拆分以加快该过程。它还可用于自动化任务并减少所需的体力劳动量。

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
