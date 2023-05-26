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
            'pin_memory': True,
            # 是一个用来指定数据加载器的参数，它可以指定你的数据加载器是否将数据加载到内存中
        } if args.is_cuda else {}

        val_dataset = datasets.ImageFolder(
            args.val_dir,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.ToTensor()
                # transforms.Normalize(opt.data_mean, opt.data_std)
            ])
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


        # 优化器
        self.optimizer = optim.AdamW(
            # AdamW是一种优化器，它是Adam优化器的变体，用于深度学习模型的训练。
            # 它的主要优点是它可以更有效地调整学习率，从而更快地收敛到最优解。AdamW优化器可以帮助模型更快地收敛，从而提高模型的准确性和性能。
            self.model.parameters(),
            lr=args.lr
        )

        # 损失函数
        self.loss_function = nn.CrossEntropyLoss()

    def test(self):
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
        test_acc, test_loss = worker.test()
        if epoch > args.save_station:
            save_dir = args.directory + '%s-epochs-%d-model-test-acc-%.3f-loss-%.6f.pt' \
                       % (args.model, epoch, test_acc, test_loss)
            torch.save(worker.model, save_dir)
