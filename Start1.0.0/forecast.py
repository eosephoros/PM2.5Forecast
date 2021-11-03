# encoding=utf-8
# @Author: Zhang Zhiyang
# @Date:   10-23-20
# @Email:  415573678@qq.com
# @Last modified by:   Zhang Zhiyang
# @Last modified time: 10-26-20
import csv
import pandas as pd
import numpy as np

index = ['AMB_TEMP', 'CH4', 'CO', 'NMHC', 'NO', 'NO2',
         'NOx', 'O3', 'PM10', 'PM2.5', 'RAINFALL', 'RH', 'SO2', 'THC', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR']
# 读取模型
w = np.load('model.npy')
# 读取训练数据
test_text = pd.read_csv("datas/test.csv", header=None)
test_text.drop([0, 1], axis=1, inplace=True)
test_text.columns = range(0, 9)
test_text[test_text == 'NR'] = '0'
test_text = test_text.astype('float')
print(test_text)
list_features = []
for i in range(240):
    test_column = test_text[0 + 18 * i:18 + 18 * i]
    test_column.index = index
    array = test_column[0].tolist()
    for n in range(8):
        array.extend(test_column[n + 1].tolist())
    array.extend("1")
    list_features.append(array)
# 开始预测并写入数据
filename = "datas/predict1.csv"
text = open(filename, "w+")
text.truncate(0)
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["id", "value"])
for i in range(240):
    list_features = np.asarray(list_features)
    labels = 0
    for j in range(163):
        labels += float(list_features[i][j].astype('float')) * w[j]
    s.writerow(["id", i, labels])
