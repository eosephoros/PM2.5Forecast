# encoding=utf-8
# @Author: Zhang Zhiyang
# @Date:   10-23-20
# @Email:  415573678@qq.com
# @Last modified by:   Zhang Zhiyang
# @Last modified time: 10-24-20
import math
import pandas as pd
import numpy as np
import time
import random


class train_model(object):
    def __init__(self):
        # 定义两个初识变量
        self.learning_step = 0.01  # 学习率
        self.max_iteration = 20000  # 最大迭代次数

    def train(self, features, labels):
        self.w = [0.01] * (len(features[0]) + 1)  # 权值加个1，相当于给b的值
        prev_gra = 0
        time = 0  # 已经迭代的次数
        x = np.concatenate((np.ones((features.shape[0], 1)), features), axis=1)
        # 开始训练
        while time < self.max_iteration:
            y_ = []
            y = labels
            y_.extend(np.dot(x, self.w))
            # 计算损失函数及梯度
            Loss = np.asarray(y_) - y
            gradient = 2 * np.dot(x.T, Loss)
            # 进行学习率的更新并计算得出新的权重w
            prev_gra += gradient ** 2
            ada = np.sqrt(prev_gra)
            self.w -= self.learning_step * gradient / ada
            # 每200次输出当前的方差
            if time % 200 == 0:
                sum_loss = 0
                sum_loss += np.dot(Loss, Loss)
                print(time, "!!!", math.sqrt(sum_loss / 5652))
            time += 1
        return self.w


if __name__ == '__main__':
    # 读出数据
    train_features = np.load("train_x.npy", allow_pickle=True)
    train_labels = np.load("train_y.npy", allow_pickle=True)

    model = train_model()
    w = model.train(train_features, train_labels)
    print(w)
    np.save('model.npy', w)
