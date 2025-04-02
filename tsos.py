import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from sklearn.metrics import confusion_matrix, classification_report

# 参数配置
config = {
    "data_dir_train": r"D:\Code\vision\data\cub\train",
    "data_dir_test": r"D:\Code\vision\data\cub\val",
    "batch_size": 64,
    "num_workers": 2,
    "num_epochs": 30,
    "num_classes": 200,
    "input_size": 224,
    "lr": 0.001,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
}

# 数据增强和加载
def prepare_dataloaders():
    # 训练集数据增强
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(config['input_size']),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 测试集预处理
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config['input_size']),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 创建数据集
    train_dataset = datasets.ImageFolder(
        config['data_dir_train'],
        transform=train_transforms
    )
    
    test_dataset = datasets.ImageFolder(
        config['data_dir_test'],
        transform=test_transforms
    )

    # 创建数据加载器
    dataloaders = {
        'train': DataLoader(train_dataset,
                          batch_size=config['batch_size'],
                          shuffle=True,
                          num_workers=config['num_workers'],
                          pin_memory=True),
        'test': DataLoader(test_dataset,
                         batch_size=config['batch_size'],
                         shuffle=False,
                         num_workers=config['num_workers'],
                         pin_memory=True)
    }
    return dataloaders

# 模型初始化函数
def initialize_model(model_name):
    model = None
    if model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, config['num_classes'])
    elif model_name == "vgg19":
        model = models.vgg19(pretrained=True)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, config['num_classes'])
    elif model_name == "densenet161":
        model = models.densenet161(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, config['num_classes'])
    elif model_name == "efficientnet_b4":
        model = models.efficientnet_b4(pretrained=True)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, config['num_classes'])
    
    if model:
        model = model.to(config['device'])
        print(f"Initialized {model_name} with {sum(p.numel() for p in model.parameters()):,} parameters")
        return model
    else:
        raise ValueError(f"Unknown model name: {model_name}")

# 训练验证流程
def train_model(model, dataloaders, criterion, optimizer, scheduler, model_name):
    since = time.time()
    best_acc = 0.0
    history = {
        'train_loss': [],
        'test_loss': [],
        'train_acc': [],
        'test_acc': []
    }

    for epoch in range(config['num_epochs']):
        print(f'Epoch {epoch+1}/{config["num_epochs"]}')
        print('-' * 10)

        # 每个epoch包含训练和验证阶段
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(config['device'])
                labels = labels.to(config['device'])

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if phase == 'train':
                scheduler.step()
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['test_loss'].append(epoch_loss)
                history['test_acc'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 保存最佳模型
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), f'best_{model_name}.pth')
                print(f'New best model saved at epoch {epoch+1} with test acc {best_acc:.4f}')

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'Best test Acc: {best_acc:.4f}')
    
    return history

# 可视化函数
def visualize_results(all_history):
    plt.figure(figsize=(15, 10))
    
    # 准确率曲线
    plt.subplot(2, 1, 1)
    for model_name, history in all_history.items():
        plt.plot(history['test_acc'], '-o', label=f'{model_name} Test')
        plt.plot(history['train_acc'], '--', label=f'{model_name} Train')
    plt.title('Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)

    # 损失曲线
    plt.subplot(2, 1, 2)
    for model_name, history in all_history.items():
        plt.plot(history['test_loss'], '-o', label=f'{model_name} Test')
        plt.plot(history['train_loss'], '--', label=f'{model_name} Train')
    plt.title('Loss Comparison')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()

# 主函数
def main():
    dataloaders = prepare_dataloaders()
    models_to_train = ['resnet50', 'vgg19', 'densenet161', 'efficientnet_b4']
    all_history = {}

    for model_name in models_to_train:
        print(f'\n{"="*40}')
        print(f'Training {model_name}')
        print(f'{"="*40}')
        
        model = initialize_model(model_name)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        history = train_model(model, dataloaders, criterion, optimizer, scheduler, model_name)
        all_history[model_name] = history
        
    visualize_results(all_history)

if __name__ == '__main__':
    main()