import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

# 加载单个batch
def load_cifar_batch(filename):
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
    data = batch['data']
    labels = batch['labels']
    return data, labels

# 加载数据集
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

# 定义数据集类
class CIFAR10Dataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx].reshape(3, 32, 32).astype(np.float32) / 255.0
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
