import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import time  # 添加计时模块


# 定义SENet模块
class SELayler(nn.Module):
    def __init__(self, channel, reduction=16):
        # 初始化函数，继承自nn.Module
        super(SELayler, self).__init__()
        # 使用自适应平均池化层，将输入的每个通道的二维空间尺寸压缩到1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 定义一个全连接层序列，包含三个部分：
        # 1. 第一个线性层，将输入的通道数压缩到原来的1/reduction
        # 2. ReLU激活函数，增加非线性
        # 3. 第二个线性层，将压缩后的通道数恢复到原来的通道数
        # 4. Sigmoid激活函数，将输出限制在0到1之间
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 前向传播函数
        # 获取输入x的尺寸，b为batch size，c为通道数，_为宽度和高度
        b, c, _, _ = x.size()
        # 使用自适应平均池化层对输入x进行池化操作，得到每个通道的全局平均特征
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # 将全局平均特征与输入x相乘，得到SE模块的输出
        return x * y.expand_as(x)


# 修改后的Residual Block，集成了SENet
class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanse, planse, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, reduction=16):
        super(ResidualBlock, self).__init__()
        # 定义Residual Block的各个部分
        self.conv1 = nn.Conv2d(inplanse, planse, kernel_size=3, stride=stride, padding=1, bias=False)
        # 第一个卷积层，输入通道数为inplanse，输出通道数为planse，卷积核大小为3x3，步长为stride，填充为1，不使用偏置
        self.bn1 = nn.BatchNorm2d(planse)
        # 第一个批量归一化层，输入通道数为planse
        self.relu = nn.ReLU(inplace=True)
        # ReLU激活函数，inplace=True表示在原地进行操作，节省内存
        self.conv2 = nn.Conv2d(planse, planse, kernel_size=3, stride=1, padding=1, bias=False)
        # 第二个卷积层，输入通道数和输出通道数均为planse，卷积核大小为3x3，步长为1，填充为1，不使用偏置
        self.bn2 = nn.BatchNorm2d(planse)
        # 第二个批量归一化层，输入通道数为planse
        self.se = SELayler(planse, reduction)
        # SE模块，用于提升特征表达，输入通道数为planse，reduction为压缩比例
        self.downsample = downsample
        # 下采样操作，如果需要下采样，则传入相应的操作，否则为None
        self.stride = stride

        # 步长，用于卷积操作中的步长参数

    def forward(self, x):
        # 前向传播函数，接收输入x
        identity = x

        # 保存输入x到identity，用于后续的残差连接
        out = self.conv1(x)
        # 通过第一个卷积层conv1对输入x进行卷积操作
        out = self.bn1(out)
        # 对卷积后的输出进行批量归一化（Batch Normalization）
        out = self.relu(out)

        # 对归一化后的输出进行ReLU激活
        out = self.conv2(out)
        # 通过第二个卷积层conv2对激活后的输出进行卷积操作
        out = self.bn2(out)
        # 对第二次卷积后的输出进行批量归一化
        out = self.se(out)

        # 对归一化后的输出进行SE模块（Squeeze-and-Excitation）处理，提升特征表达
        if self.downsample is not None:
            # 如果存在下采样操作
            identity = self.downsample(x)

            # 对输入x进行下采样，使尺寸与out匹配
        out += identity
        # 将处理后的out与原始输入identity相加，实现残差连接
        out = self.relu(out)

        # 对相加后的输出进行ReLU激活
        return out


# Resnet 架构定义，这里以ResNet-18为例，并集成SENet 从头开始搭建SENet
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=200):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []  # 确保 layers 始终被初始化
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)  # 确保始终返回 nn.Sequential

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
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x












# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
train_dataset = datasets.ImageFolder(root=rf'D:\Code\vision\data\cub/train', transform=transform)
val_dataset = datasets.ImageFolder(root=rf'D:\Code\vision\data\cub/val', transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
num_classes = len(train_dataset.classes)

# 加载预训练的 ResNet18 模型
resnet18 = models.resnet18(weights='IMAGENET1K_V1')
resnet18.fc = nn.Linear(resnet18.fc.in_features, num_classes)

resnetnew = ResNet(ResidualBlock, [2, 2, 2, 2])


model = resnetnew




# 将模型移动到 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 200

for epoch in range(num_epochs):
    start_time = time.time()  # 记录当前 epoch 开始时间

    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # 计算训练时间
    epoch_time = time.time() - start_time

    # 打印训练信息（带 \n 分隔符）
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    print(f"训练损失: {running_loss / len(train_loader):.4f}")
    print(f"训练时间: {epoch_time:.2f} 秒")

    # 验证模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(f"验证准确率: {val_accuracy:.4f}%")

# 保存模型
torch.save(model.state_dict(), 'cub_resnet18.pth')
print(f"模型已保存为 cub_resnet18.pth\n")