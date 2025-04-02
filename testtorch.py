# -*- coding: utf-8 -*-
import os
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

# 了解图像分类骨干网络，如AlexNet、ResNet、SENet、Transformer、Mamba等

# 打印信息
print("\n=== 信息 ===")
print(f"PyTorch版本: {torch.__version__}")
print(f"TorchVision版本: {torchvision.__version__}")
print(f"检测到的CUDA设备数量: {torch.cuda.device_count()}")
print(f"当前CUDA设备: {torch.cuda.current_device()}")
print(f"设备名称: {torch.cuda.get_device_name(0)}")
print(f"CUDA版本: {torch.version.cuda}")

# 导入模型
from torchvision import models
from torchvision import models,transforms

# 模型定义
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()

        # 特征提取层
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2),  # 输入通道数为3，输出通道数为64，卷积核大小为11，步长为4，填充为2
            nn.ReLU(inplace=True),  # 最大池化层，池化核大小为3，步长为2，填充为0
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(64, 192, kernel_size=5,stride=1 ,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),  # 最大池化层，池化核大小为3，步长为2，填充为0
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),  # 输入通道数为192，输出通道数为384，卷积核大小为3，步长为1，填充为1
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),  # 输入通道数为384，输出通道数为256，卷积核大小为3，步长为1，填充为1
            nn.ReLU(inplace=True),  # 激活函数为ReLU
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )

        #自适应池化层
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))  # 自适应池化层，将特征图的大小调整为6x6
        # 分类层
        self.classifier = nn.Sequential(
            nn.Dropout(),  # Dropout层，用于防止过拟合
            nn.Linear(256 * 6 * 6, 4096),  # 全连接层，输入特征数为256*6*6，输出特征数为4096
            nn.ReLU(inplace=True),  # 全连接层，输入特征数为4096，输出特征数为4096
            nn.Dropout(),  # Dropout层，用于防止过拟合
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    # 前向传播
    def forward(self, x):
        # 将输入x传入特征提取层
        x = self.features(x)
        # 对特征提取层的结果进行全局平均池化
        x = self.avgpool(x)
        # 将池化后的结果展平为一维向量
        x = torch.flatten(x, 1)
        # 将展平后的向量传入分类器进行分类
        x = self.classifier(x)
        # 返回分类结果
        return x



if __name__ == '__main__':

    # 定义训练数据集的路径
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 加载训练数据集
    transform_augmented = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.Grayscale(num_output_channels=3),  # 转换为 3 通道
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    #下载CIFAR10数据集
    train_data = torchvision.datasets.MNIST(root="./data",train=True,transform=transform_augmented,
                                            download=True)
    test_data = torchvision.datasets.MNIST(root="./data",train=False,transform=transform_augmented,
                                            download=True)

    train_data_size = len(train_data)
    test_data_size = len(test_data)
    print("训练集大小： {}".format(train_data_size))
    print("测试集大小： {}".format(test_data_size))

    #dataloder进行数据集的加载
    trainloader = DataLoader(train_data,batch_size=128)
    testloader = DataLoader(test_data,batch_size=128)

    # 定义resnet模型
    resnet = models.resnet18(weights='IMAGENET1K_V1').to(device)
    resnet.fc = nn.Sequential(nn.Linear(resnet.fc.in_features,10),
                                nn.LogSoftmax(dim=1))

    # AlexNet模型
    Alexnet = AlexNet(num_classes=10).to(device)

    # 定义VGG模型
    vgg = models.vgg16(weights='IMAGENET1K_V1').to(device)

    # 定义resnet_50模型
    resnet_50 = models.resnet50(weights='IMAGENET1K_V1').to(device)
    resnet_50.fc = nn.Sequential(nn.Linear(resnet_50.fc.in_features,10),
                                nn.LogSoftmax(dim=1))

    # MobileNet模型
    mobilenet = models.mobilenet_v2(weights='IMAGENET1K_V1').to(device)
    mobilenet.classifier[1] = nn.Linear(mobilenet.classifier[1].in_features, 10)

    # 设置训练设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vgg.to(device)
    resnet.to(device)
    mobilenet.to(device)

    criterion = nn.CrossEntropyLoss()

    # 定义损失函数
    loss_fn = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()

    # 定义优化器
    learning_rate = 0.1
    optimizer = torch.optim.SGD(resnet.parameters(),lr=learning_rate,)

    print(optimizer)

    #设置网络训练的一些参数
    #记录训练的次数
    total_train_step = 0
    #记录测试的次数
    total_test_step = 0
    #训练的轮数
    epoch = 5

    model = Alexnet.to(device)
    # 创建 log 目录
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    # 配置日志记录器
    log_file = os.path.join(log_dir, "train-testtorch.log")
    logging.basicConfig(
        level=logging.INFO,  # 记录 INFO 及以上级别的日志
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),  # 输出到文件
            logging.StreamHandler()  # 输出到控制台
        ]
    )
    logger = logging.getLogger()


    # 加载之前保存的模型权重
    #model_path = "./model/mnist_resnet_test.pth"
    #model.load_state_dict(torch.load(model_path))
    #logger.info(f"从 {model_path} 加载模型权重")



    # 训练模型
    num_epochs = 20
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
    model_path = f"./model/mnist_resnet_test{epoch}.pth"
    torch.save(model.state_dict(), model_path)
    logger.info(f"模型已保存到 {model_path}")

    # 评估模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    logger.info(f"Test Accuracy: {accuracy:.2f}%")