# !/usr/bin/env python
# encoding: utf-8
"""
@author: Lzy
@file: distill.py
@time: 2022/2/26 14:59
@desc:
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
# from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from model import *
import torch.nn.functional as F

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
teacher_model.load_state_dict(torch.load('teacher_0.99419999.ckpt'))
student_model = Student()
student_model.to(device)

# 损失函数
loss_CEL = nn.CrossEntropyLoss()
loss_CEL.to(device)
loss_KL = nn.KLDivLoss()
loss_KL.to(device)

# 优化器
lr = 1e-3
optimzer = torch.optim.Adam(student_model.parameters(), lr=lr)

# 设置参数
epochs = 50
train_total_step = 1
test_total_step = 1
alpha = 0.5
# writer = SummaryWriter("./logs_train")

for epoch in range(epochs):
    print("-------第{}轮训练---------".format(epoch + 1))
    start_time = time.time()
    # 训练开始
    student_model.train()
    total_train_accuracy = 0
    for data in train_dataloader:
        imgs, targets = data
        imgs, targets = Variable(imgs), Variable(targets)           # 占位符，只更换值
        imgs = imgs.to(device)
        targets = targets.to(device)
        student_outputs = student_model(imgs)
        teacher_outputs = teacher_model(imgs)

        # 损失函数
        T = 1
        loss1 = loss_CEL(student_outputs, targets)      # student网络输出与标准值labels的差距，用的是交叉熵损失
        outputs_S = F.softmax(student_outputs / T, dim=1)
        outputs_T = F.softmax(teacher_outputs / T, dim=1)
        loss2 = loss_KL(outputs_S, outputs_T) * T * T
        train_total_loss = loss1 * (1 - alpha) + loss2 * alpha
        # 优化模型
        optimzer.zero_grad()    # 梯度值归零
        train_total_loss.backward()         # 反向传播计算得到每个参数的梯度值
        optimzer.step()         # 通过梯度下降执行一步参数更新

        train_total_step += 1
        if train_total_step % 100 == 0:
            print("训练次数:{}，loss:{}".format(train_total_step, train_total_loss.item()))
    end_time = time.time()
    print(end_time - start_time)

    # 测试开始
    student_model.eval()

    test_total_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs, targets = Variable(imgs), Variable(targets)
            imgs = imgs.to(device)
            targets = targets.to(device)
            # forward
            outputs = student_model(imgs)
            outputs.detach_()   # 将 outputs 从创建它的 graph 中分离，把它作为叶子节点
            # 计算loss
            loss = loss_CEL(outputs, targets)
            test_total_loss += loss.item()
            # 统计
            test_accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + test_accuracy
            # 统计混淆矩阵

    print("整体测试集上的Loss：{}".format(test_total_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy / test_len))
    if epoch == epochs - 1:
        torch.save(student_model.state_dict(), "ts_{:.8}.ckpt".format(total_accuracy / test_len))
        print("模型已保存")
