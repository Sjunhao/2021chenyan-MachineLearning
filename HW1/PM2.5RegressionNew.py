import numpy as np
import pandas as pd
import math

data = pd.read_csv(r'data/train.csv', encoding='gb18030')
data = data.iloc[:, 3:]
data[data == 'NR'] = 0
data = data.drop([i for i in range(4320) if (i-1)%18==0])
# data.to_csv(r'new.csv')
raw_data = data.to_numpy()
# print(raw_data)

# 每行为特征标签，每列存储当前小时的所有特征
mouth_data = {}
feature_nums = 17

for mouth in range(12):
    sample = np.empty([feature_nums, 480])  # 一共有18个特征，每个月有20天一天24小时20*24=480
    # 每次循环拉伸存储24个小时的数据
    for day in range(20):
        # 需要拉伸的数据=（当前月*20+当前天）*18的数据间隔
        sample[:, day * 24:(day + 1) * 24] = raw_data[feature_nums * (20 * mouth + day):feature_nums * (20 * mouth + day + 1):]
    mouth_data[mouth] = sample
# print(mouth_data)

x = np.empty([12 * (20 * 24 - 9), feature_nums * 9], dtype=float)  # 因为每个月20天不连续，标签有12个月*（20天*24小时-9小时）
y = np.empty([12 * (20 * 24 - 9), 1], dtype=float)

for mouth in range(12):
    for day in range(20):
        for hour in range(24):
            # 一个月只有20天的数据是连续的，所以每个月要从20天14小时处截断进入下一个月
            if day == 19 and hour > 14:
                continue
            x[mouth * 471 + day * 12 + hour, :] = mouth_data[mouth][:, day * 24 + hour:day * 24 + hour + 9].reshape(1,
                                                                                                                    -1)
            y[mouth * 471 + day * 12 + hour, 0] = mouth_data[mouth][9, day * 24 + hour + 9]

# 做标准化，统一量纲
mean_x = np.mean(x, axis=0)
std_x = np.std(x, axis=0)
for i in range(len(x)):
    for j in range(len(x[0])):
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]


# 随机划分训练集和验证集
def split_train(x, y, test_ratio):
    np.random.seed(43)  # 设定初始化种子
    shuffled_indices = np.random.permutation(len(x))
    test_set_size = int(len(x) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return x[train_indices], x[test_indices], y[train_indices], y[test_indices]


train_x, vali_x, train_y, vali_y = split_train(x, y, 0.2)

#添加一行对应bias项
dim = train_x.shape[1] + 1
w = np.zeros([dim, 1])
train_x = np.concatenate((np.ones([train_x.shape[0], 1]), train_x), axis=1).astype(float)

lr = 10
adagrad = np.zeros([dim, 1])
eps = 0.0005
reg_rate = 0.011

reg_mat = np.concatenate((np.array([0]),np.ones([9*feature_nums])),axis=0)

# for T in range(30000):
#     loss = np.sqrt(np.sum(np.power(train_x.dot(w) - train_y, 2)) / len(train_x))
#     if T % 500 == 0:
#         print("%d,loss:%f" % (T, loss))
#     adagrad = (train_x.T.dot(train_x.dot(w)  - train_y)) / (loss * len(train_x))
#     adagrad += adagrad ** 2
#     w = w - lr * adagrad / (np.sqrt(adagrad) + eps)

for T in range(500000):
    loss = np.sum(np.power(train_x.dot(w) - train_y, 2)) / (len(train_x) * 2)
    if T % 5000 == 0:
        print("T=%d,loss:%f" % (T, loss))
        lr /= 1.1
    gradient = 2 * (train_x.T.dot(train_x.dot(w) - train_y))
    # gradient = 2 * (train_x.T.dot(train_x.dot(w) - train_y)) / (loss * len(train_x))
    adagrad += gradient ** 2
    w -= lr * gradient / (np.sqrt(adagrad) + eps)
np.save('weights.npy', w)
