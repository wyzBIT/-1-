import torch
import torch.optim as optim
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from data_loader import load_cifar10_data, CIFAR10Dataset
from model import SimpleCNN
from train import train_model, evaluate_model, plot_metrics
import gradio as gr

# 配置超参数
config = {
    'learning_rate': 0.001,
    'batch_size': 128,
    'optimizer': 'adam',
    'num_epochs': 1
}

cifar10_dir = 'data'

# CIFAR-10 类别名称
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
]

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

# 画出损失函数和准确率的曲线
def plot_training_curves(train_losses, train_accuracies):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', color='b', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, marker='o', color='g', label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    return 'training_curves.png'


# Gradio 接口函数
def visualize_training_and_classification(image, epochs):
    # 预处理图像
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    image = nn.functional.interpolate(image, size=(32, 32))  # 确保图像大小为 32x32

    # 重置网络
    net = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=config['learning_rate'])

    # 训练模型
    train_losses, train_accuracies = train_model(net, train_loader, criterion, optimizer, device, num_epochs=epochs)
    test_accuracy = evaluate_model(net, test_loader, device)

    # 绘制训练曲线
    curves_path = plot_training_curves(train_losses, train_accuracies)

    # 分类图像
    net.eval()
    with torch.no_grad():
        output = net(image)
        _, predicted = torch.max(output, 1)
        label_idx = predicted.item()
        label_name = class_names[label_idx]  # 映射到类别名称

    return curves_path, label_name, test_accuracy


# 定义 Gradio 输入输出
image_input = gr.Image(type="numpy", label="Upload Image")
epochs_input = gr.Slider(minimum=1, maximum=50, value=10, label="Epochs")
curves_output = gr.Image(label="Training Curves")
label_output = gr.Label(label="Classification Result")

# 创建 Gradio 接口
iface = gr.Interface(
    fn=visualize_training_and_classification,
    inputs=[image_input, epochs_input],
    outputs=[curves_output, label_output],
    live=True  # 实时更新
)

# 启动接口
iface.launch()
