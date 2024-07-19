import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import load_cifar10_data, CIFAR10Dataset
from model import SimpleCNN
from train import train_model, evaluate_model, plot_metrics

# 配置超参数
config = {
    'learning_rate': 0.001,
    'batch_size': 128,
    'optimizer': 'adam',
    'num_epochs': 2
}

cifar10_dir = 'data/cifar-10-batches-py'

# 加载数据
(train_data, train_labels), (test_data, test_labels) = load_cifar10_data(cifar10_dir)
train_dataset = CIFAR10Dataset(train_data, train_labels)
test_dataset = CIFAR10Dataset(test_data, test_labels)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

# 初始化网络
net = SimpleCNN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)

# 打印训练配置
print(f"Training Configuration:")
print(f"Learning Rate: {config['learning_rate']}")
print(f"Batch Size: {config['batch_size']}")
print(f"Optimizer: {config['optimizer']}")
print(f"Number of Epochs: {config['num_epochs']}")

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
if config['optimizer'] == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=config['learning_rate'])
elif config['optimizer'] == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=config['learning_rate'], momentum=0.9)

# 训练模型
train_losses, train_accuracies = train_model(net, train_loader, criterion, optimizer, device, num_epochs=config['num_epochs'])

# 评估模型
evaluate_model(net, test_loader, device)

# 绘制训练曲线
plot_metrics(train_losses, train_accuracies)
