import torch
import torchvision

from minis_resnetse import ResNet, ResidualBlock
import torchvision.transforms as transforms
from PIL import Image


if __name__ == '__main__':
    # 加载模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet(ResidualBlock, [2, 2, 2, 2]).to(device)
    model.load_state_dict(torch.load("./model/mnist_resnet.pth", map_location=device))
    model.eval()
    print("模型已加载！")

    # 预处理方法，确保与训练时的 transform 相同
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # 将灰度图转换为 3 通道
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 加载 MNIST 数据集中的单张图片
    mnist_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    image, label = mnist_dataset[0]  # 取第一张测试集图片
    image = image.unsqueeze(0).to(device)  # 增加 batch 维度

    # 进行推理
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    print(f"预测类别: {predicted.item()}，真实类别: {label}")

