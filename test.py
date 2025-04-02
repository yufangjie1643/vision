#%% 导入PyTorch库，用于深度学习任务
import torch
# 导入torchvision库，用于计算机视觉任务
import torchvision
# 从torchvision库中导入transforms模块，用于图像预处理
import torchvision.transforms as transforms
# 从torchvision库中导入models模块，用于获取预训练模型
from torchvision import models
# 导入torch.nn模块，用于构建神经网络
import torch.nn as nn
# 导入torch.optim模块，用于优化算法
import torch.optim as optim
# 导入模型权重
from torchvision.models import VGG16_Weights, ResNet18_Weights, MobileNet_V2_Weights

# 导入tqdm库，用于显示训练进度条
from tqdm import tqdm
# 导入matplotlib.pyplot库，用于绘图
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np




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

    def __init__(self, block, layers, num_classes=10):
        # 初始化ResNet类，block为残差块类型，layers为各层的块数，num_classes为分类数，默认为10
        super(ResNet, self).__init__()
        # 调用父类（nn.Module）的初始化方法
        self.inplanes = 64
        # 输入通道数
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

            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

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
    plt.show()

if __name__ == '__main__':
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # 将图像大小调整为224x224
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # 加载训练集和测试集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                            shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                            shuffle=False, num_workers=4)

    transform_augmented = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset_augmented = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_augmented)
    trainloader_augmented = torch.utils.data.DataLoader(trainset_augmented, batch_size=32,
                                            shuffle=True, num_workers=4)

    # 3.模型定义与初始化(直接调用与重新搭建)
        # 定义resnet模型
    resnet = models.resnet18(weights='IMAGENET1K_V1').eval()
    for param in resnet.parameters():
        param.requires_grad = False #False：冻结模型的参数，也就是采用该模型已经训练好的原始参数。只需要训练我们自己定义的Linear层
    #保持in_features不变，修改out_features=10
    resnet.fc = nn.Sequential(nn.Linear(resnet.fc.in_features,10),
                                nn.LogSoftmax(dim=1))


    # 定义VGG模型
    vgg16 = models.vgg16(weights='IMAGENET1K_V1').eval()
    for param in vgg16.parameters():
        param.requires_grad = False #False：冻结模型的参数，也就是采用该模型已经训练好的原始参数。只需要训练我们自己定义的Linear层
    vgg16.classifier[6] = nn.Linear(4096, 10)

    # MobileNet模型
    mobilenet = models.mobilenet_v2(weights='IMAGENET1K_V1').eval()
    for param in mobilenet.parameters():
        param.requires_grad = False #False：冻结模型的参数，也就是采用该模型已经训练好的原始参数。只需要训练我们自己定义的Linear层
    mobilenet.classifier[1] = nn.Linear(mobilenet.classifier[1].in_features, 10)

    # 设置训练设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vgg.to(device)
    resnet.to(device)
    mobilenet.to(device)


    # 训练模型
    criterion = nn.CrossEntropyLoss()
    optimizer_vgg = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)
    optimizer_resnet = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)
    optimizer_mobilenet = optim.SGD(mobilenet.parameters(), lr=0.001, momentum=0.9)

    vgg_train_loss = []
    resnet_train_loss = []
    mobilenet_train_loss = []

    for epoch in range(10):
        print('Epoch {}/{}'.format(epoch+1, 10))

        #Training VGG16
        vgg16.train()
        running_loss = 0.0
        for i, data in enumerate(tqdm(trainloader, desc="Training VGG16")):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer_vgg.zero_grad()
            outputs = vgg16(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_vgg.step()

            running_loss += loss.item()
        avg_loss = running_loss / len(trainloader)
        vgg_train_loss.append(avg_loss)

        #Training ResNet
        resnet.train()
        running_loss = 0.0
        for i, data in enumerate(tqdm(trainloader, desc="Training ResNet")):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer_resnet.zero_grad()
            outputs_resnet = resnet(inputs)
            loss_resnet = criterion(outputs_resnet, labels)
            loss_resnet.backward()
            optimizer_resnet.step()

            running_loss += loss_resnet.item()
        avg_loss = running_loss / len(trainloader)
        resnet_train_loss.append(avg_loss)

        #Training MobileNet
        mobilenet.train()
        running_loss = 0.0
        for i, data in enumerate(tqdm(trainloader, desc="Training MobileNet")):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer_mobilenet.zero_grad()
            outputs_mobilenet = mobilenet(inputs)
            loss_mobilenet = criterion(outputs_mobilenet, labels)
            loss_mobilenet.backward()
            optimizer_mobilenet.step()

            running_loss += loss_mobilenet.item()
        avg_loss = running_loss / len(trainloader)
        mobilenet_train_loss.append(avg_loss)



    test_model(vgg16, "VGG16")
    test_model(resnet, "ResNet")
    test_model(mobilenet, "MobileNet")