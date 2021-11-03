# encoding=utf-8
# @Author: Zhang Zhiyang
# @Date:   10-23-20
# @Email:  415573678@qq.com
# @Last modified by:   Zhang Zhiyang
# @Last modified time: 10-24-20

import sys
import csv
import pandas as pd
import numpy as np
import random
import time

if __name__ == '__main__':
    print('Reading Data')
    # 定义所需数据格式的行名
    index = ['AMB_TEMP', 'CH4', 'CO', 'NMHC', 'NO', 'NO2',
             'NOx', 'O3', 'PM10', 'PM2.5', 'RAINFALL', 'RH', 'SO2', 'THC', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR']
    # 读取训练数据
    train_text = pd.read_csv("datas/train.csv", encoding='big5')
    train_column = train_text["0"]
    print(train_column)
    train_text.drop(['日期', '測站', '測項'], axis=1, inplace=True)
    # 取出前18行作为初始数据
    train_column = train_text[0:18]
    Data = train_column
    Data.index = index
    Data.columns = range(0, 24)

    # 将后续数据依次加入Data中形成所需要的数据格式
    for i in range(1, 240):
        print(i)
        train_column = train_text[0 + i * 18:18 + i * 18]
        train_column.index = index
        train_column.columns = range(0 + i * 24, 24 + i * 24)
        list1 = [Data.T, train_column.T]
        Data = pd.concat(list1).T
    # 表数据处理，改格式成float，降水量NR改成0
    Data[Data == 'NR'] = '0'
    Data = Data.astype('float')
    print(Data[0].tolist())

    # 取出特征及标签
    list_features = []
    list_result = []
    for i in range(12):
        print('第', i + 1, '个月')
        for j in range(471):  # 除去最后的9小时得到471组数据
            num = 0 + i * 480 + j  # 初始数字是第num列
            array = Data[num].tolist()
            for n in range(8):
                array.extend(Data[num + n + 1].tolist())
            list_features.append(array)
            label = Data[num + 9].iloc[9]
            list_result.append(label)

    # 保存数据
    train_features = np.asarray(list_features)
    print(train_features.shape)  # 5652
    train_labels = np.asarray(list_result)
    print(train_features)
    np.save('train_x', train_features)
    np.save('train_y', train_labels)
