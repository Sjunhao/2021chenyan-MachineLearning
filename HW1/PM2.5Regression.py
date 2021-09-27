#!/usr/bin/env python
# coding: utf-8

# # 一、实验说明
# 参考https://github.com/Iallen520/lhy_DL_Hw/blob/master/hw1_regression.ipynb

# 给定训练集train.csv，测试集test.csv，要求根据前9个小时的空气检测情况预测第10个小时的PM2.5含量
# 
# 

# 要求：
# 
# 现在给出的代码是这个问题的baseline，请大家：
# 
# 1. 给出代码注释；
# 
# 2. 想办法优化算法的结果。
# 
# 3.有条件的同学，请使用这个地址参加对应的kaggle比赛 https://www.kaggle.com/c/ml2020spring-hw1/overview 
# 将参赛的结果页面截图粘贴到提交的代码notebook最下方。
# 
# 4. 如果无法参加kaggle比赛，请修改代码，将数据集的trainning data的10%切分为测试集。进行测试。注意训练集和测试集数据不能共用。
# 
# 

# # 二、训练集介绍

# 1. CSV文件，包含台湾丰原地区240天的气象观测资料，取每个月前20天的数据做训练集，每月后10天数据用于测试；
# 2. 每天的监测时间点为0时，1时......到23时，共24个时间节点；
# 3. 每天的检测指标包括CO、NO、PM2.5、PM10等气体浓度，是否降雨、刮风等气象信息，共计18项；
# 

# - train.csv部分数据展示

# ![jupyter](./img/traindata_eg.png)

# In[30]:


import sys
import pandas as pd
import numpy as np


# # 三、数据预处理

# 浏览数据可知，数据中存在一定量的空数据NR，且多存在于RAINFALL一项。RAINFALL表示当天对应时间点是否降雨，有降雨值为1，无降雨值为NR，类似于布尔变量。因此将空数据NR全部补为0即可

# In[31]:


# data = pd.read_csv('/home/aistudio/data/data27964/train.csv', encoding = 'big5' ) # 读取结果的结构是DataFrame
data = pd.read_csv(r'data/train.csv', encoding = 'big5') # 读取结果的结构是DataFrame


# - Pandas里主要数据结构包含DataFrame（二维表），如上打印结果，有行有列。但标准说法行（索引），列（标签）

# In[32]:


# panda里利用iloc选取数据，从0开始。iloc（行，列）
# 当前选取从第三列开始的所有数据
data = data.iloc[:, 3:]
data[data=='NR'] = 0
data = pd.DataFrame(data,dtype=np.float)
data_corr = data.corr()
print(data_corr)
# data


# In[33]:


raw_data = np.array(data) # DataFrame转换成numpy数组


# In[34]:


print(raw_data.shape)


# In[35]:


print(raw_data)


# # 四、特征提取

# ## （1）按月份来处理数据
# - 针对每20天来说，信息维度[18, 480] (18个feature，20*24=480)
# - 将原始的数据按照每个月来划分，重组成12个 [18,480]

# In[14]:


month_data = {}  # key: month  value: data
for month in range(12):
    sample = np.empty([18, 480])  # 创建一个空的【18， 480】数组
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
    month_data[month] = sample


# In[ ]:


# 以第一个月为例
# print(month_data[0])


# ## （2）扩充数据集，获取更好的训练效果
# - 根据实验要求，需要用连续9个时间点的数据预测第10个时间点的PM2.5。 而每个月采取的是前20天连续的数据，可以扩展成480小时的连续数据；
# - 具体做法，每个月的第一天的0-8时的数据作为训练数据，9时的数据作标签y；9-17的数据作一个data，18时的数据作标签y.....以此推，每个月480小时，有480-9= 471个data，故此时总数据471 * 12 个；而每个data是18*9

# In[15]:


x = np.empty([12 * 471, 18 * 9], dtype = float)
y = np.empty([12 * 471, 1], dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            # reshape将矩阵重整为新的行列数，参数-1代表自动推断,这里去掉了18*9的二维属性，
            # 转而以一维序列代替，一维序列的顺序本身可以隐含其时序信息
            x[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1) 
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9] #value
print(x)
print(y)


# In[16]:


mean_x = np.mean(x, axis = 0) # 求均值， aix=0表示沿每列计算
std_x = np.std(x, axis = 0) # 标准差
for i in range(len(x)): 
    for j in range(len(x[0])): 
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j] # 所有属性归一化，避免使数据的某些特征形成主导作用


# In[ ]:


x


# # 损失函数
# - 采用预测值与标签y之间的平均欧时距离来衡量预测的准确程度
# ![jupyter](./img/Loss.png)
# - num = 471*12， 乘 1/2 是为了在后续求梯度过程中保证梯度系数为1，方便计算

# # 学习率更新

# 为了在不影响模型效果的前提下提高学习速度，可以对学习率进行实时更新：即让学习率的值在学习初期较大，之后逐渐减小。这里采用比较经典的adagrad算法来更新学习率。

# In[17]:


dim = x.shape[1] + 1 
w = np.zeros(shape = (dim, 1 )) #empty创建的数组，数组中的数取决于数组在内存中的位置处的值，为0纯属巧合？
x = np.concatenate((np.ones((x.shape[0], 1 )), x) , axis = 1).astype(float) 

#初始化学习率(163个参数，163个200)和adagrad
learning_rate = np.array([[200]] * dim)
adagrad_sum = np.zeros(shape = (dim, 1 ))
 
#没有隐藏层的网络
for T in range(10001):
    if(T % 500 == 0 ):
        print("T=",T)
        print("Loss:",np.sum((x.dot(w) - y)**2)/ x.shape[0] /2) #最小二乘损失
        print((x.dot(w) - y)**2)
    gradient = 2 * np.transpose(x).dot(x.dot(w)-y) #损失的导数x*(yh-h)
    adagrad_sum += gradient ** 2
    w = w - learning_rate * gradient / (np.sqrt(adagrad_sum) + 0.0005)

np.save('weight.npy',w)


# In[ ]:


w.shape


# In[ ]:


x.shape


# # 使用模型预测

# In[19]:


# 同处理训练集数据一样
# testdata = pd.read_csv('/home/aistudio/data/data27964/test.csv', header = None, encoding = 'big5')
testdata = pd.read_csv(r'D:\PyCharmWorkplace\2021chenyan-MachineLearning\HW1\data\test.csv', header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
test_data = test_data.copy() # 为防止pandas出错：A value is trying to be set on a copy of a slice from a DataFrame.

test_data[test_data == 'NR'] = 0


# In[21]:


test_data = np.array(test_data)
test_data.shape


# In[22]:



w = np.load('weight.npy')
 
test_x = np.empty(shape = (240, 18 * 9),dtype = float)
 

for i in range(240):
    test_x[i,:] = test_data[18 * i : 18 * (i+1),:].reshape(1,-1) 
 
for i in range(test_x.shape[0]):        ##Normalization
    for j in range(test_x.shape[1]):
        if not std_x[j] == 0 :
            test_x[i][j] = (test_x[i][j]- mean_x[j]) / std_x[j]
 
test_x = np.concatenate((np.ones(shape = (test_x.shape[0],1)),test_x),axis = 1).astype(float)


# In[23]:


test_x.shape


# In[24]:


w.shape


# In[25]:


w = np.load('weight.npy')
ans_y = np.dot(test_x, w)


# In[26]:


import csv
with open('submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)


# In[ ]:





# In[ ]:




