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
# 测试输出
# print(init_train_data)
#      0      1      2      3       4     ...    5755    5756    5757    5758    5759
# 0   14.00  14.00  14.00  13.00   12.00  ...   14.00   13.00   13.00   13.00   13.00
# 1    1.80   1.80   1.80   1.80    1.80  ...    1.80    1.80    1.80    1.80    1.80
# ...
# [18 rows x 5760 columns]

# 3.处理训练集

