# !/usr/bin/env python
# encoding: utf-8
"""
@author: Lzy
@file: teacher.py
@time: 2022/2/27 15:43
@desc:
"""

import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
# from torch.utils.tensorboard import SummaryWriter
from model import *

# 设备
device = torch.device("cuda:0")

# 准备数据集
from torch.utils.data import DataLoader

train_data = torchvision.datasets.MNIST(root='./MNIST_dataset/', train=True,
                                        transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.MNIST(root='./MNIST_dataset/', train=False,
                                       transform=torchvision.transforms.ToTensor(), download=True)

train_len = len(train_data)
test_len = len(test_data)
print("train长度:{}".format(train_len))
print("test长度:{}".format(test_len))

# 加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
teacher_model = Teacher()
teacher_model.to(device)

# 损失函数
loss_CEL = nn.CrossEntropyLoss()
loss_CEL.to(device)
loss_KL = nn.KLDivLoss()
loss_KL.to(device)

# 优化器
lr = 1e-3
optimzer = torch.optim.Adam(teacher_model.parameters(), lr=lr)

# 设置参数
epochs = 50
train_total_step = 1
test_total_step = 1
# writer = SummaryWriter("./logs_train")

for epoch in range(epochs):
    print("-------第{}轮训练---------".format(epoch + 1))
    start_time = time.time()
    # 训练开始
    teacher_model.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = teacher_model(imgs)
        loss = loss_CEL(outputs, targets)

        # 优化模型
        optimzer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimzer.step()

        train_total_step += 1
        if train_total_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数；{}，loss：{}".format(train_total_step, loss.item()))

    # 测试开始
    teacher_model.eval()
    test_total_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = teacher_model(imgs)
            loss = loss_CEL(outputs, targets)
            test_total_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print("整体测试集上的Loss：{}".format(test_total_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy / test_len))

    if epoch == epochs - 1:
        torch.save(teacher_model.state_dict(), "teacher_{}.ckpt".format(epoch))
        print("模型已保存")
