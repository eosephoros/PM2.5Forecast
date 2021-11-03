import pandas as pd
import numpy as np
import csv

# 1. 读取数据，big5是针对于文档中存在繁体字的编码
train_data = pd.read_csv('datas/train.csv', usecols=range(2, 27), encoding='big5')
row = train_data['測項'].unique()

# 2.训练数据规整化
# 定义18维空列表，每一个维度处理一种污染物
new_train_data = pd.DataFrame(np.zeros([18, 24 * 240]))
# 计算维度
n = 0
# 遍历18类污染物，将数据处理为 [18, 24*240]
for i in row:
    # 依照污染物名取出相同的行
    train_data1 = train_data[train_data['測項'] == i]
    train_data1.drop(['測項'], axis=1, inplace=True)
    # 格式处理，赋值到 new_train_data
    train_data1 = np.array(train_data1)
    train_data1[train_data1 == 'NR'] = '0'
    train_data1 = train_data1.astype('float')
    train_data1 = train_data1.reshape(5760, 1)
    train_data1 = train_data1.T
    new_train_data.loc[n] = train_data1
    # 下一类污染物处理
    n += 1

train_array = np.array(new_train_data).astype(float)

# 3. 处理训练集
# 训练样本features集合
X_train = []
# 训练样本目标PM2.5集合
y_train = []
for i in range(new_train_data.shape[1] - 9):
    # 每次取9个小时的数据作训练集
    # 每次数据遍历每行前9个数据，全部加入到训练集中，18 X 9
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

    # 取本次第10个小时的 PM2.5 的值作为训练的真实值
    y = int(train_array[9, i + 9])
    # 将样本分别存入X_train、y_train中
    y_train.append(y)
X_train = np.array(X_train)
y_train = np.array(y_train)

# 4.实现线性回归
# 训练轮数
epoch = 2000
# 开始训练
# 更新参数，训练模型
# 偏置值初始化
bias = 0
# 权重初始化
weights = np.ones(18 * 9)
# 初始学习率
learning_rate = 1
# 用于存放偏置值的梯度平方和
bg2_sum = 0
# 用于存放权重的梯度平方和
wg2_sum = np.zeros(18 * 9)

for i in range(epoch):
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

    # adagrad
    bg2_sum += b_g ** 2
    wg2_sum += w_g ** 2
    # 更新权重和偏置
    bias -= learning_rate / bg2_sum ** 0.5 * b_g
    weights -= learning_rate / wg2_sum ** 0.5 * w_g

    # 每训练200轮，输出一次在训练集上的损失
    if i % 200 == 0:
        loss = 0
        for j in range(len(X_train)):
            loss += (y_train[j] - weights.dot(X_train[j]) - bias) ** 2
        print('after {} epochs, the loss on train data is:'.format(i), loss / len(X_train))

# 5.存储模型
# save model
np.save('model_weight.npy', weights)
np.save('model_bias.npy', bias)
# read model
w = np.load('model_weight.npy')
b = np.load('model_bias.npy')

# 6.读取测试数据
# 6.1 处理测试集
# 读取数据，big5是针对于文档中存在繁体字的编码
test_data = pd.read_csv('datas/test.csv', header=None, usecols=range(2, 11), encoding='big5')

# 将测试数据中 NR 转化为 0
for i in range(test_data.shape[0]):
    for j in range(2, 11):
        if test_data.loc[i][j] == 'NR':
            test_data.loc[i][j] = '0'

# 6.2 测试样本features集合
X_test = []
for i in range(int(test_data.shape[0] / 18)):
    X_temp = np.ones(18 * 9)
    # 记录
    count = 0
    for j in range(18 * i, 18 * (i + 1)):
        for k in range(2, 11):
            X_temp[count] = test_data.loc[j][k]
            count += 1
    # 将样本分别存入X_train中
    X_test.append(X_temp)
X_test = np.array(X_test).astype(float)

pre_list = []
for i in range(len(X_test)):
    pre = w.dot(X_test[i]) - b
    pre_list.append(pre)
    print('id_', i, ' ', pre)

# 6.3 保存预测结果到文件中
# 读入文件数据
sampleSubmission = open('datas/sampleSubmission.csv', 'r')
reader = csv.reader(sampleSubmission)
rows = [rowTemp for rowTemp in reader]

# 重新打开，准备写入
newfile = open('datas/sampleSubmission.csv', 'w', newline='')

writer = csv.writer(newfile)
# 写入列名
writer.writerow(rows[0])
for i in range(len(rows) - 1):
    # 将预测值写入
    rows[i + 1][1] = pre_list[i]
    writer.writerow(rows[i + 1])

newfile.close()
sampleSubmission.close()

# 8.测试结果
# 测试结果样本features集合
y_test = pd.read_csv('datas/predict.csv', usecols=range(1, 2))
y_test = np.array(y_test).astype(float)

loss = 0
for i in range(len(X_test)):
    loss += (y_test[i] - w.dot(X_test[i]) - b) ** 2
print(loss)
print('The loss on test data is:', loss / len(X_test))
