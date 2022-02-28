# !/usr/bin/env python
# encoding: utf-8
"""
@author: Lzy
@file: model.py
@time: 2022/2/26 16:20
@desc:
"""
import torch
from torch import nn
import torch.nn.functional as F


class Student(nn.Module):

    def __init__(self):
        super(Student, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.maxpool1 = nn.MaxPool2d(2, 1)
        self.fc1 = nn.Linear(3750, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(F.relu(x))
        # torch.reshape(x, (1, -1))
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        return x
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill(1)
                m.biae.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()


class Teacher(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.pooling1 = nn.Sequential(nn.MaxPool2d(2, 2))
        self.fc = nn.Sequential(nn.Linear(6272, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pooling1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pooling1(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    teacher = Student()
    # print(teacher)
    input = torch.ones((64, 1, 28, 28))
    output = teacher(input)
    print(output.shape)