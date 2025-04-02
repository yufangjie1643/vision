import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 定义AlexNet模型
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()

        # 特征提取层
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(64, 192, kernel_size=5,stride=1 ,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )
        #自适应池化层
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # 分类层
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # 将图像大小调整为224x224
        transforms.Grayscale(num_output_channels=3),  # 转换为 3 通道
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True,num_workers=2)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=False,num_workers=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AlexNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

   # 创建 log 目录
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    # 配置日志记录器
    log_file = os.path.join(log_dir, "train_Alexnet.log")
    logging.basicConfig(
        level=logging.INFO,  # 记录 INFO 及以上级别的日志
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),  # 输出到文件
            logging.StreamHandler()  # 输出到控制台
        ]
    )
    logger = logging.getLogger()


    # 训练模型
    num_epochs = 10
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
    model_path = "./model/mnist_Alex.pth"
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
   