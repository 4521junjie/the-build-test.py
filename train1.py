import argparse
import time
import json
import os
from argparse import Namespace

from tqdm import tqdm
from models.ResNet34_update import *
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
    parser.add_argument('--model', type=str, default='ResNet34')

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
                transforms.RandomResizedCrop(224),
                transforms.ToTensor()
            ])
        )
        val_dataset = datasets.ImageFolder(
            args.val_dir,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
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
            # self.val_loader = DataLoader 是一个 Python 库，用于加载和组织机器学习模型的数据。它有助于快速有效地将数据加载到模型中，并使其更易于处理和分析。
            # 它可用于从各种源加载数据，例如 CSV 文件、数据库和其他源。它还有助于将数据组织成批次，从而更轻松地训练模型。此外，它还可用于执行数据增强，这有助于提高模型的准确性。
            dataset=val_dataset,
            # 数据集是以特定方式组织和格式化的数据集合。在这种情况下，val_dataset是以对特定目的有用的方式组织和格式化的数据集合。
            batch_size=args.test_batch_size,

            shuffle=False,

            **kwargs
        )

        # 挑选神经网络、参数初始化
        net = ResNet34()
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

    # 训练与验证
    for epoch in range(1, args.epochs + 1):
        worker.train(epoch)
        val_acc, val_loss = worker.val()
        if epoch > args.save_station:
            save_dir = args.directory + '%s-epochs-%d-model-val-acc-%.3f-loss-%.6f.pt' \
                       % (args.model, epoch, val_acc, val_loss)
            torch.save(worker.model, save_dir)
