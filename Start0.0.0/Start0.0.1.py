import pandas as pd
import numpy as np
import csv

# 1.读取训练数据
train_data = pd.read_csv('datas/train.csv', usecols=range(2, 27), encoding='big5')
row = train_data['測項'].unique()
# 测试输出
# print(row)
# 输出结果为['AMB_TEMP' 'CH4' 'CO' 'NMHC' 'NO' 'NO2' 'NOx' 'O3' 'PM10' 'PM2.5'
#  'RAINFALL' 'RH' 'SO2' 'THC' 'WD_HR' 'WIND_DIREC' 'WIND_SPEED' 'WS_HR']

# 2.训练数据初始化
#   定义污染物列表
init_train_data = pd.DataFrame(np.zeros([18, 24 * 240]))
# 测试输出
# init_train_data_test = pd.DataFrame(np.zeros([3,4]))
# print(init_train_data_test)
#      0    1    2    3
# 0  0.0  0.0  0.0  0.0
# 1  0.0  0.0  0.0  0.0
# 2  0.0  0.0  0.0  0.0
# 计算维度
n = 0
# 遍历18类污染物，讲数据处理为[18,24*240]
for i in row:
    # 依照污染物名取出相同的行
    train_data1 = train_data[train_data['測項'] == i]
    train_data1.drop(['測項'], axis=1, inplace=True)

    train_data1 = np.array(train_data1)
    train_data1[train_data1 == 'NR'] = '0'
    train_data1 = train_data1.astype('float')
    train_data1 = train_data1.reshape(5760, 1)
    train_data1 = train_data1.T
    init_train_data.loc[n] = train_data1
    # 下一类污染物
    n += 1

train_array = np.array(init_train_data).astype(float)
# 测试输出
# print(init_train_data)
#      0      1      2      3       4     ...    5755    5756    5757    5758    5759
# 0   14.00  14.00  14.00  13.00   12.00  ...   14.00   13.00   13.00   13.00   13.00
# 1    1.80   1.80   1.80   1.80    1.80  ...    1.80    1.80    1.80    1.80    1.80
# ...
# [18 rows x 5760 columns]

# 3.处理训练集
# 训练样本features集合
X_train = []
# 训练样本目标PM2.5集合
y_train = []
for i in range(init_train_data.shape[1] - 9):
    # 每次取9个小时的数据作为训练集
    # 每次数据遍历每行前九个数据，全部加入到训练集中，18*19
    X_temp = np.ones(18 * 9)
    # 记录
    count = 0
    for j in range(18):
        x = train_array[j, i:i + 9]
        for k in range(9):
            X_temp[count] = x[k]
            count += 1
    # 将样本分别存入X_train中
    X_train.append(X_temp)

    # 取本次第10个小时的PM2.5的值作为训练的真实值
    y = int(train_array[9, i + 9])
    # 取样本分别存入X_train,y_train
    y_train.append(y)

# 测试输出
# print(type(y_train))
# 输出为：list
X_train = np.array(X_train)
y_train = np.array(y_train)
# 测试输出
# print(type(y_train))
# 输出为:numpy.ndarray

# 4.实现线性回归
# 训练轮数
epoch = 2000
# 更新参数、训练模型
# 初始化偏置值
bias = 0
# 设置权重
weights = np.ones(18 * 9)
# 初始学习率
learning_rate = 1
# 存放偏置值的梯度平方和
bg2_sum = 0
# 存放权重的平方和
wg2_sum = 0
for i in range(epoch):
    print(i)
    b_g = 0
    w_g = np.zeros(18 * 9)
    # 在所有数据上计算Loss_label的梯度
    for j in range(len(X_train)):
        b_g += (y_train[j] - weights.dot(X_train[j]) - bias) * (-1)
        for k in range(18 * 9):
            w_g[k] += (y_train[j] - weights.dot(X_train[j]) - bias) * (-X_train[j, k])
        # 求平均
    b_g /= len(X_train)
    w_g /= len(X_train)

    # adagrade
    bg2_sum += b_g ** 2
    wg2_sum += w_g ** 2

    # 更新权重和偏重
    bias -= learning_rate / bg2_sum ** 0.5 * b_g
    weights -= learning_rate / wg2_sum ** 0.5 * w_g

    # 每训练200轮，输出一次在训练集上的损失
    if i % 200 == 0:
        loss = 0
        for j in range(len(X_train)):
            loss += (y_train[j] - weights.dot(X_train[j]) - bias) ** 2
        print('after {} epochs, the loss on train data is : '.format(i), loss / len(X_train))

# 5.存储模型
np.save('model_weight.npy', weights)
np.save('model_bias.npy', bias)

# 读取模型
w = np.load('model_weight.npy')
b = np.load('model_bias.npy')
