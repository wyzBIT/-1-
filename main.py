import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary

# CIFAR-10 dataset directory
cifar10_dir = 'data/cifar-10-batches-py'

# Hyperparameters and settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 10
batch_size = 64
learning_rate = 0.001

# Function to load a single batch
def load_cifar_batch(filename):
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
    data = batch['data']
    labels = batch['labels']
    return data, labels

# Load all training batches
def load_cifar10_data(data_dir):
    train_data = []
    train_labels = []
    for i in range(1, 6):
        data, labels = load_cifar_batch(os.path.join(data_dir, f'data_batch_{i}'))
        train_data.append(data)
        train_labels.append(labels)
    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)
    test_data, test_labels = load_cifar_batch(os.path.join(data_dir, 'test_batch'))
    return (train_data, train_labels), (test_data, test_labels)

# Custom dataset class
class CIFAR10Dataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data  # 数据集中的图像数据
        self.labels = labels  # 图像对应的标签
        self.transform = transform  # 可选的图像变换

    def __len__(self):
        return len(self.data)  # 返回数据集的大小

    def __getitem__(self, idx):
        # 获取图像数据并重新调整形状，标准化到[0, 1]
        image = self.data[idx].reshape(3, 32, 32).astype(np.float32) / 255.0
        label = self.labels[idx]  # 获取图像对应的标签
        if self.transform:
            image = self.transform(image)  # 如果提供了图像变换，应用变换
        # 将图像数据和标签转换为 PyTorch 张量
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Load data
(train_data, train_labels), (test_data, test_labels) = load_cifar10_data(cifar10_dir)

# 创建数据集对象
train_dataset = CIFAR10Dataset(train_data, train_labels)
test_dataset = CIFAR10Dataset(test_data, test_labels)
# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Initialize the network
net = SimpleCNN().to(device)

# 打印网络结构
summary(net, (3, 32, 32))

def train_model(net, train_loader, criterion, optimizer, num_epochs):
    """
    训练神经网络模型。

    Args:
    - net (nn.Module): 要训练的神经网络模型。
    - train_loader (DataLoader): 训练数据集的数据加载器，用于获取训练数据的小批量。
    - criterion: 损失函数，用于计算模型预测与真实标签之间的误差。
    - optimizer: 优化器，用于更新模型的参数，例如随机梯度下降（SGD）或 Adam。
    - num_epochs (int): 训练的轮数。
    """
    net.train()  # 将模型设为训练模式
    for epoch in range(num_epochs):  # 遍历每个 epoch
        running_loss = 0.0  # 初始化累计损失为 0
        for i, (inputs, labels) in enumerate(train_loader):  # 遍历每个 mini-batch
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到 GPU
            optimizer.zero_grad()  # 清零梯度
            outputs = net(inputs)  # 前向传播，计算模型输出
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新模型参数
            running_loss += loss.item()  # 累加损失值
            if i % 100 == 99:  # 每处理完 100 个 mini-batch 打印一次损失
                print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss / 100}')
                running_loss = 0.0  # 重置累计损失
    print('Finished Training')  # 训练结束

# 定义损失函数为交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 定义优化器为 Adam 优化器，用于更新神经网络模型 net 的参数
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# Train the model
train_model(net, train_loader, criterion, optimizer, num_epochs)

def evaluate_model(net, test_loader):
    """
    评估神经网络模型在测试集上的准确率。

    Args:
    - net (nn.Module): 已训练的神经网络模型。
    - test_loader (DataLoader): 测试数据集的数据加载器，用于获取测试数据的小批量。
    """
    net.eval()  # 将模型设为评估模式，这会影响到一些层的行为，如 Dropout 层会关闭

    correct = 0  # 初始化预测正确的样本数为 0
    total = 0  # 初始化总样本数为 0

    with torch.no_grad():  # 禁用梯度计算，因为在评估阶段我们不需要进行参数更新
        for inputs, labels in test_loader:  # 遍历测试集中的每个 mini-batch
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到 GPU
            outputs = net(inputs)  # 将输入数据输入到模型中进行预测
            _, predicted = torch.max(outputs, 1)  # 获取预测值中概率最大的类别
            total += labels.size(0)  # 累加样本总数，labels.size(0) 表示当前 mini-batch 的样本数
            correct += (predicted == labels).sum().item()  # 计算预测正确的样本数

    # 输出在测试集上的准确率
    print(f'Accuracy on test set: {100 * correct / total:.2f}%')

# Evaluate the model
evaluate_model(net, test_loader)
