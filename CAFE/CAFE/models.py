"""
共享模型定义 —— 训练和推理脚本共用，避免代码重复。
包含：my_MaxPool2d、BasicBlock、ResNet、Mask、supervisor、Model
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import clip as clip_module

# ==================== 自定义池化层 ====================
class my_MaxPool2d(nn.Module):
    """在维度 1 上做 max pooling（自定义 transpose 实现）。"""
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,  
                 return_indices=False, ceil_mode=False):  
        # dilation就是卷积核内部参数之间的间距  比如第一层 1 1 1 -> 1 0 1 0 1
        super(my_MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices  # 是否返回最大值的位置索引
        self.ceil_mode = ceil_mode  # 控制输出尺寸的计算方式，True表示向上取整，False表示向下取整

    def forward(self, input):
        input = input.transpose(3, 1)  # 转置维度 方便池化
        input = F.max_pool2d(input, self.kernel_size, self.stride,
                             self.padding, self.dilation, self.ceil_mode,
                             self.return_indices)
        input = input.transpose(3, 1).contiguous()  # 确保张量连续存储
        return input


# ==================== ResNet-18 组件 ====================
class BasicBlock(nn.Module):
    """ResNet-18 的基础残差块（expansion=1）。"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        #  downsample是一个布尔值，表示是否需要下采样操作 便于调整尺寸
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        if downsample:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                             stride=stride, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None

        self.downsample = downsample

    def forward(self, x):
        i = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            i = self.downsample(i)

        x += i
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    """ResNet-18 特征提取器（最后一层之前的部分）。"""
    def __init__(self, block, n_blocks, channels, output_dim):
        # 参数含义 选取的网络(比如resnet18 34) 残差结构的数目 每层输出通道数(比如64 128) 全连接层的输出维度
        super().__init__()
        self.in_channels = channels[0]
        assert len(n_blocks) == len(channels) == 4  # 确保参数正确

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.get_resnet_layer(block, n_blocks[0], channels[0])
        self.layer2 = self.get_resnet_layer(block, n_blocks[1], channels[1], stride=2)
        self.layer3 = self.get_resnet_layer(block, n_blocks[2], channels[2], stride=2)
        self.layer4 = self.get_resnet_layer(block, n_blocks[3], channels[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, output_dim)

    def get_resnet_layer(self, block=BasicBlock, n_blocks=[2, 2, 2, 2],
                         channels=[64, 128, 256, 512], stride=1):
        layers = []
        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False
        layers.append(block(self.in_channels, channels, stride, downsample))
        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))
        self.in_channels = block.expansion * channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)
        return x, h


# ==================== 监督损失模块 ====================
def Mask(nb_batch, device='cuda'):
    # nb_batch: 当前有多少张图片
    """
    生成随机二值掩码。
    512 维特征分为 7 个 chunk（每个73维，最后一组74维含补偿），
    每个 chunk 中随机屏蔽 10 个位置（设为 0），其余为 1。
    这样强制模型不能只依赖少数维度做判别。
    """
    bar = []
    for i in range(7):
        foo = [1] * 63 + [0] * 10
        if i == 6:
            foo = [1] * 64 + [0] * 10   # 最后一组多1个有效位
        np.random.shuffle(foo)  # 随机打乱
        bar += foo
    bar = [bar for _ in range(nb_batch)]   # 复制 B 份 → (B, 512)，每张图独立掩码
    # 每张图的掩码独立，不能共享
    bar = np.array(bar).astype("float32")  # list → np.ndarray
    bar = bar.reshape(nb_batch, 512, 1, 1) # (B, 512, 1, 1) 适配卷积特征图形状
    bar = torch.from_numpy(bar)            # numpy → torch.Tensor
    bar = bar.to(device)                   # 搬到 GPU/CPU
    bar = Variable(bar)                    # 包装为 Variable（兼容旧版写法，等同 Tensor）
    return bar


def supervisor(x, targets, cnum, device='cuda'):
    """
    监督分支损失（两部分组成）：
      - loss_1: 对掩码后特征做 max pooling + CrossEntropy，鼓励单个 chunk 内判别能力
      - loss_2: 特征多样性正则，鼓励各 chunk 均匀贡献（避免某一维度独大）
    """
    branch = x
    branch = branch.reshape(branch.size(0), branch.size(1), 1, 1)
    branch = my_MaxPool2d(kernel_size=(1, cnum), stride=(1, cnum))(branch)
    branch = branch.reshape(branch.size(0), branch.size(1),
                            branch.size(2) * branch.size(3))
    loss_2 = 1.0 - 1.0 * torch.mean(torch.sum(branch, 2)) / cnum

    mask = Mask(x.size(0), device=device)
    branch_1 = x.reshape(x.size(0), x.size(1), 1, 1) * mask
    branch_1 = my_MaxPool2d(kernel_size=(1, cnum), stride=(1, cnum))(branch_1)
    branch_1 = branch_1.view(branch_1.size(0), -1)
    loss_1 = nn.CrossEntropyLoss()(branch_1, targets)

    return [loss_1, loss_2]


# ==================== CAFE 完整模型 ====================
class Model(nn.Module):
    """
    CAFE (CLIP-Assisted Facial Expression) 模型。

    架构：
      1. ResNet-18（MSCeleb 预训练）提取面部特征
      2. CLIP ViT-B-32（冻结）并行编码图像
      3. sigmoid(ResNet特征) 逐元素门控 CLIP 特征
      4. 全连接层输出 7 类评分

    训练模式下额外输出 MC_loss（监督分支损失）。
    """
    def __init__(self, clip_model, num_classes=7, resnet_pretrained_path=None, device='cpu'):
        super(Model, self).__init__()

        self.clip_model = clip_model
        self.device = device

        res18 = ResNet(block=BasicBlock, n_blocks=[2, 2, 2, 2],
                       channels=[64, 128, 256, 512], output_dim=1000)

        if resnet_pretrained_path is not None:
            msceleb_model = torch.load(resnet_pretrained_path, map_location=device)
            state_dict = msceleb_model['state_dict']
            res18.load_state_dict(state_dict, strict=False)

        self.features = nn.Sequential(*list(res18.children())[:-2])
        self.features2 = nn.Sequential(*list(res18.children())[-2:-1])

        fc_in_dim = list(res18.children())[-1].in_features  # = 512
        self.fc = nn.Linear(fc_in_dim, num_classes)

    def forward(self, x, targets=None, phase='train'):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(x)

        x = self.features(x)
        x = self.features2(x)
        x = x.view(x.size(0), -1)

        MC_loss = None
        if phase == 'train':
            MC_loss = supervisor(
                image_features * torch.sigmoid(x),
                targets, cnum=73, device=self.device
            )

        x = image_features * torch.sigmoid(x)
        out = self.fc(x)

        if phase == 'train':
            return out, MC_loss
        else:
            return out
