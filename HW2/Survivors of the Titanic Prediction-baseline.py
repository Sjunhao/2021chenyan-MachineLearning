#!/usr/bin/env python
# coding: utf-8

# 以下代码参考自网络
# * https://blog.csdn.net/aicanghai_smile/article/details/79234172
# * https://www.jianshu.com/p/9a5bce0de13f
# * https://www.kaggle.com/c/titanic/data
# 经过少量修改

# 数据介绍：
# * PassengerId:乘客ID
# * Survived:是否获救
# * Pclass:乘客等级
# * Name:乘客姓名
# * Sex:性别
# * Age:年龄
# * SibSp:堂兄弟妹个数
# * Parch:父母与小孩个数
# * Ticket:船票信息
# * Fare:票价
# * Cabin:客舱
# * Embarked:登船港口
# 

# ## 实验要求
# 请给部分代码加上注释

# In[1]:


# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. This directory will be recovered automatically after resetting environment. 
# get_ipython().system('ls /home/aistudio/data')


# In[2]:


# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. All changes under this directory will be kept even after reset. Please clean unnecessary files in time to speed up environment loading.
# get_ipython().system('ls /home/aistudio/work')


# In[3]:


# get_ipython().system('pip install seaborn -i https://mirrors.aliyun.com/pypi/simple')


# In[4]:


import numpy as np
import pandas as pd
from IPython.display import display
# get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[5]:


#使用read_csv将train.csv和test.csv分别读到train和test两个dataframe里


display(train.head(n=1),test.head(n=1))


# In[6]:


train.info()
test.info()
train.describe()


# In[7]:


#使用countplot画出train中不同Pclass下生存和死亡人数的柱状图


# ## 请写出上一句代码所得的柱状图你自己的分析结论

# In[8]:


#使用countplot画出train中不同性别下生存和死亡人数的柱状图


# ## 请写出上一句代码所得的柱状图你自己的分析结论

# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

# In[9]:


#以下代码不能在一次运行中反复执行，因为x已经被改过了
train['Age']=train['Age'].map(lambda x: 'yes' if 0<x<100 else 'no')   #有年龄信息和没有年龄信息是否和死亡与否有关
sns.countplot(x="Age",hue="Survived",data=train)


# In[10]:


train = pd.read_csv('/home/aistudio/data/data6374/train.csv')
                                                           #画出小提琴图（violinplot）以分析生存和死亡的人的年龄分布


# In[11]:


#给以下代码加注释
                                                                     #使用lambda表达式将age的数值变为child（<12）,youth(<30),adult(<60),old(<75),tooold(>=75),null(其他)）


# In[12]:


#使用countplot画出堂兄妹个数与是否生存的关系柱状图，横坐标为堂兄妹个数，纵坐标为生存和死亡的人数


# In[13]:


train['SibSp']=train['SibSp'].map(lambda x: 'small' if x<1 else 'middle' if x<3 else 'large')        


# In[14]:


sns.countplot(x="Parch", hue="Survived", data=train)


# In[15]:


train['Parch']=train['Parch'].map(lambda x: 'small' if x<1 else 'middle' if x<4 else 'large')


# In[16]:


sns.violinplot(x='Survived',y='Fare',data=train)


# ## 请写出上一句代码所得的风琴图你自己的分析结论

# In[17]:


#train = pd.read_csv('/home/aistudio/data/data6374/train.csv')
train['Fare']=train['Fare'].map(lambda x:np.log(x+1)) #由于部分x为0，因此要+1   请写出这个地方为什么用log，不用log是什么样的
sns.violinplot(x='Survived',y='Fare',data=train)


# In[18]:


#使用lambda表达式将Fare的数值变为poor（<2.5）,rich(其他)）


# In[19]:


#使用lambda表达式将Cabin改为'yes'(type为str)，‘no’（type为其他））
sns.countplot(x="Cabin", hue ="Survived", data =train)


# In[20]:


sns.countplot(x="Embarked",hue="Survived",data=train)


# In[21]:


train.dropna(axis=0,inplace=True)                         #请写出这句的注释
train.info()


# In[23]:


labels=train['Survived']
features = train.drop(['Survived','PassengerId','Name','Ticket'],axis=1)          #请写出这句的注释和这么操作的原因


# In[24]:


features.info()


# In[25]:


features = pd.get_dummies(features)   #请给出本句代码的注释，并解释为什么这么做
encoded = list(features.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))


# In[ ]:



#对test中的'Age','SibSp'，'Parch'特征进行分段分类



#均值补齐'Fare'特征值并作对数转换和分类，下面缺了两句
test.Fare.fillna(test['Fare'].mean(), inplace=True)
 

#按'Cabin'是否缺损对test中的Cabin处理


#删除不需要的特征并进行独热编码



encoded = list(test.columns)
print ("{} total features after one-hot encoding.".format(len(encoded)))


# In[ ]:


get_ipython().system('pip install xgboost -i https://mirrors.aliyun.com/pypi/simple')


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score,roc_auc_score
from time import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier


# In[ ]:


def fit_model(alg,parameters):
    X=features
    y=labels
    scorer = make_scorer(roc_auc_score)
    grid=GridSearchCV(alg,parameters,scoring=scorer,cv=5)   #自学GridSearchCV及其他自动调参方法，了解他们的优缺点并在此简单列出
    start=time()
    grid=grid.fit(X,y)
    end=time()
    t=round(end-start,3)
    print(grid.best_params_)
    print ('searching time for {} is {} s'.format(alg.__class__.__name__,t)) #输出搜索时间
    return grid


# In[ ]:


#列出需要使用的算法
alg1=DecisionTreeClassifier(random_state=29)
alg2=SVC(probability=True,random_state=29)  #由于使用roc_auc_score作为评分标准，需将SVC中的probability参数设置为True
alg3=RandomForestClassifier(random_state=29)
alg4=AdaBoostClassifier(random_state=29)
alg5=KNeighborsClassifier(n_jobs=-1)
alg6=XGBClassifier(random_state=29,n_jobs=-1)


# In[ ]:


#列出需要调整的参数范围
parameters1={'max_depth':range(1,10),'min_samples_split':range(2,10)}
parameters2 = {"C":range(1,20), "gamma": [0.05,0.1,0.15,0.2,0.25]}
parameters3_1 = {'n_estimators':range(10,200,10)}
parameters3_2 = {'max_depth':range(1,10),'min_samples_split':range(2,10)}  #搜索空间太大，分两次调整参数
parameters4 = {'n_estimators':range(10,200,10),'learning_rate':[i/10.0 for i in range(5,15)]}
parameters5 = {'n_neighbors':range(2,10),'leaf_size':range(10,80,20)  }
parameters6_1 = {'n_estimators':range(10,200,10)}
parameters6_2 = {'max_depth':range(1,10),'min_child_weight':range(1,10)}
parameters6_3 = {'subsample':[i/10.0 for i in range(1,10)], 'colsample_bytree':[i/10.0 for i in range(1,10)]}#搜索空间太大，分三次调整参数


# 请把模型调用部分的代码改成循环形式

# In[ ]:


clf1=fit_model(alg1,parameters1)


# In[ ]:


clf2=fit_model(alg2,parameters2)


# In[39]:


clf3_m1=fit_model(alg3,parameters3_1)


# In[40]:


alg3=RandomForestClassifier(random_state=29,n_estimators=180)
clf3=fit_model(alg3,parameters3_2)


# In[ ]:


clf4=fit_model(alg4,parameters4)


# In[ ]:


clf5=fit_model(alg5,parameters5)


# In[ ]:


clf6_m1=fit_model(alg6,parameters6_1)


# In[ ]:


alg6=XGBClassifier(n_estimators=140,random_state=29,n_jobs=-1)
clf6_m2=fit_model(alg6,parameters6_2)


# In[ ]:


alg6=XGBClassifier(n_estimators=140,max_depth=4,min_child_weight=5,random_state=29,n_jobs=-1)
clf6=fit_model(alg6,parameters6_3)


# In[ ]:


def save(clf,i):
    pred=clf.predict(test)
    sub=pd.DataFrame({ 'PassengerId': Id, 'Survived': pred })
    sub.to_csv("res_tan_{}.csv".format(i), index=False)


# In[ ]:



i=1
for clf in [clf1,clf2,clf3,clf4,clf5,clf6]:
    save(clf,i)
    i=i+1

