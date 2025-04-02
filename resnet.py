#%% 导入PyTorch库，用于深度学习任务
import logging
import os
import torch
# 导入torchvision库，用于计算机视觉任务
import torchvision
# 从torchvision库中导入transforms模块，用于图像预处理
import torchvision.transforms as transforms
# 从torchvision库中导入models模块，用于获取预训练模型
from torchvision import models,datasets
# 导入torch.nn模块，用于构建神经网络
import torch.nn as nn
# 导入torch.optim模块，用于优化算法
import torch.optim as optim

# 导入tqdm库，用于显示训练进度条
from tqdm import tqdm
# 导入matplotlib.pyplot库，用于绘图
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
from torch.utils.data import DataLoader
# 了解图像分类骨干网络，如AlexNet、ResNet、SENet、Transformer、Mamba等


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
    
def test_model(model, name):
    # 设置模型为评估模式，关闭dropout和batchnorm
    model.eval()
    # 初始化正确预测数和总预测数
    correct = 0
    total = 0
    # 初始化存储所有标签和预测结果的列表
    all_labels = []
    all_preds = []

    # 在不计算梯度的情况下进行预测
    with torch.no_grad():
        # 遍历测试数据集
        for data in testloader:
            # 将图像和标签移动到指定设备（如GPU）
            images, labels = data[0].to(device), data[1].to(device)
            # 获取模型输出
            outputs = model(images)
            # 获取预测结果（最大概率的类别）
            _, predicted = torch.max(outputs.data, 1)
            # 更新总预测数
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    recall = 100 * recall_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Accuracy of {name} : {accuracy:.2f}%")
    print(f"Recall of {name} : {recall:.2f}%")

    # 绘制混淆矩阵
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=trainset.classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix of {name}")
    # 保存图像到指定路径，例如 'confusion_matrix.png'
    plt.savefig(f"confusion_matrix_of_{name}.png")
    plt.show()

# 自定义数据集类
class DAGMDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_path_list = glob(os.path.join(self.root_dir, mode, '*', '*.bmp'))

        # 动态生成类别到标签的映射
        self.classes = sorted(list(set([img_path.split(os.sep)[-2] for img_path in self.img_path_list])))
        self.class_to_label = {cls: idx for idx, cls in enumerate(self.classes)}

        # 打印找到的文件路径
        print(f"Found {len(self.img_path_list)} images in {mode} mode.")
        print(f"Classes: {self.classes}")
        print(f"Class to label mapping: {self.class_to_label}")

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        img = Image.open(img_path).convert('RGB') # 确保图像是 RGB 格式
        label = self.class_to_label[img_path.split(os.sep)[-2]] # 根据路径获取类别

        if self.transform:
            img = self.transform(img)

        return img, label
#%%
if __name__ == '__main__':

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])



   # 加载数据集
    trainset = datasets.ImageFolder(root=rf'D:\Code\vision\data\cub/train', transform=transform)

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    # 加载测试集
    testset = datasets.ImageFolder(root=rf'D:\Code\vision\data\cub/val', transform=transform)

    testloader = DataLoader(testset, batch_size=32, shuffle=False)
    
    # 初始化模型、损失函数和优化器
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, 200)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    # 创建 log 目录
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    # 配置日志记录器
    log_file = os.path.join(log_dir, "cub_resnetse.log")
    logging.basicConfig(
        level=logging.INFO,  # 记录 INFO 及以上级别的日志
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),  # 输出到文件
            logging.StreamHandler()  # 输出到控制台
        ]
    )
    logger = logging.getLogger()




    #%% 训练模型
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    
        avg_loss = running_loss / len(trainloader)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")


    # 训练完成后保存模型
    model_path = "./model/cub_resnet_yfj.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"模型已保存到 {model_path}")

    # 评估模型
    test_model(model,"resnetse_cub")
   



