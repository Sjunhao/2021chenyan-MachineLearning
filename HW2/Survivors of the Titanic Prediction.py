import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 数据读入
train = pd.read_csv(r'Data/train.csv')
test = pd.read_csv(r'Data/test.csv')

print(train.info())
# 画图
# train中不同Pclass下生存和死亡人数的柱状图
# sns.countplot(x=train['Pclass'],hue=train['Survived'])
# plt.show()
# train中不同性别下生存和死亡人数的柱状图
# sns.countplot(x=train['Sex'], hue=train['Survived'])
# plt.show()
"""
结论：在属性Pclass下，Pclass=1存活几率高，Pclass=3死亡机率更大
    在属性Sex下，man更容易存活
    所以这两个属性与是否存活有较强的相关性
"""
### 对训练数据做离散化处理
temp_train = train.copy()
temp_train['Age'] = train['Age'].map(lambda x: 'yes' if 0 < x < 100 else 'no')
# sns.countplot(x='Age', hue='Survived', data=temp_train)
# plt.show()
# sns.violinplot(hue='Age', x='Survived', data=temp_train)
# plt.show()
temp_train['Age'] = train['Age'].map(lambda
                                         x: 'child' if x < 12 else 'youth' if x > 30 else 'adult' if x < 60 else 'old' if x < 75 else 'tooold' if x >= 75 else 'null')
# 堂兄妹个数与是否生存的关系柱状图，横坐标为堂兄妹个数，纵坐标为生存和死亡的人数
# sns.countplot(x='SibSp',hue='Survived',data=temp_train)
# plt.show()
temp_train['SibSp'] = train['SibSp'].map(lambda x: 'small' if x < 1 else 'middle' if x < 3 else 'large')
temp_train['Parch'] = train['Parch'].map(lambda x: 'small' if x < 1 else 'middle' if x < 4 else 'large')
## Fare差距太大不容易离散化所以log
train['Fare'] = train['Fare'].map(lambda x: np.log(x + 1))
# sns.violinplot(x='Survived',y='Fare',data=train)
# plt.show()
"""
    结论：票价高的存活几率越大
"""
temp_train['Fare'] = train['Fare'].map(lambda x: 'poor' if x < 2.5 else 'rich')
temp_train['Cabin'] = train['Cabin'].map(lambda x: 'yes' if type(x) == str else 'no')
temp_train.dropna(axis=0, inplace=True)  # 删掉NA的行
labels = temp_train['Survived']
## 删除一些无关的列和Y，减少数据量
features = temp_train.drop(['Survived', 'PassengerId', 'Name', 'Ticket'], axis=1)
### get_dummies（）可以做one-hot encoding，离散值大小没意义的时候做
features = pd.get_dummies(features)

### 对测试数据做离散化处理
test['Age'] = test['Age'].map(lambda
                                  x: 'child' if x < 12 else 'youth' if x > 30 else 'adult' if x < 60 else 'old' if x < 75 else 'tooold' if x >= 75 else 'null')
test['SibSp'] = test['SibSp'].map(lambda x: 'small' if x < 1 else 'middle' if x < 3 else 'large')
test['Parch'] = test['Parch'].map(lambda x: 'small' if x < 1 else 'middle' if x < 4 else 'large')
test['Cabin'] = test['Cabin'].map(lambda x: 'yes' if type(x) == str else 'no')
test.Fare.fillna(test['Fare'].mean(), inplace=True)
test['Fare'] = test['Fare'].map(lambda x: np.log(x + 1))
test['Fare'] = test['Fare'].map(lambda x: 'poor' if x < 2.5 else 'rich')
test.Cabin.fillna('no', inplace=True)
test.dropna(axis=0, inplace=True)
Id = test['PassengerId']
test = test.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
test = pd.get_dummies(test)
features.info()
test.info()

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, roc_auc_score
from time import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier


def fit_model(alg, parameters):
    X = features
    y = labels
    scorer = make_scorer(roc_auc_score)  # 使用ROC函数的AUC进行评价
    grid = GridSearchCV(alg, parameters, scoring=scorer, cv=5)
    """
        GridSearchCV(网格搜索):自动调参。
        GridSearchCV可以保证在指定的参数范围内找到精度最高的参数，
        但是这也是网格搜索的缺陷所在，他要求遍历所有可能参数的组合，在面对大数据集和多参数的情况下，非常耗时。
        
        参数：
        estimator：所使用的分类器，如estimator=RandomForestClassifier(min_samples_split=100,min_samples_leaf=20,max_depth=8,max_features='sqrt',random_state=10), 并且传入除需要确定最佳的参数之外的其他参数。每一个分类器都需要一个scoring参数，或者score方法。
        param_grid：值为字典或者列表，即需要最优化的参数的取值，param_grid =param_test1，param_test1 = {'n_estimators':range(10,71,10)}。
        scoring :准确度评价标准，默认None,这时需要使用score函数；或者如scoring='roc_auc'，根据所选模型不同，评价准则不同。字符串（函数名），或是可调用对象，需要其函数签名形如：scorer(estimator, X, y)；如果是None，则使用estimator的误差估计函数。scoring参数选择如下：
        cv :交叉验证参数，默认None，使用三折交叉验证。指定fold数量，默认为3，也可以是yield训练/测试数据的生成器。
        refit :默认为True,程序将会以交叉验证训练集得到的最佳参数，重新对所有可用的训练集与开发集进行，作为最终用于性能评估的最佳模型参数。即在搜索参数结束后，用最佳参数结果再次fit一遍全部数据集。
        iid:默认True,为True时，默认为各个样本fold概率分布一致，误差估计为所有样本之和，而非各个fold的平均。
        verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：偶尔输出，>1：对每个子模型都输出。
        n_jobs: 并行数，int：个数,-1：跟CPU核数一致, 1:默认值。
        pre_dispatch：指定总共分发的并行任务数。当n_jobs大于1时，数据将在每个运行点进行复制，这可能导致OOM，而设置pre_dispatch参数，则可以预先划分总共的job数量，使数据最多被复制pre_dispatch次
        
        常用方法：
        grid.fit()：运行网格搜索
        grid_scores_：给出不同参数情况下的评价结果
        best_params_：描述了已取得最佳结果的参数的组合
        best_score_：成员提供优化过程期间观察到的最好的评分
    """

    start = time()
    grid = grid.fit(X, y)
    end = time()
    t = round(end - start, 3)
    print(grid.best_params_)
    print('searching time for {} is {} s'.format(alg.__class__.__name__, t))  # 输出搜索时间
    return grid


# 列出需要使用的算法
alg1 = DecisionTreeClassifier(random_state=29)  # 决策树，给 random_state 参数赋值任意整数，都可以使模型在同一个训练集和测试集下稳定。
alg2 = SVC(probability=True, random_state=29)  # 支持向量机 ，probability决定最后是否按概率输出每种可能的概率
alg3 = RandomForestClassifier(random_state=29)  # 随机森林，n_estimators是森林中树木的数量，即基评估器的数量。n_estimators越大，模型的效果往往越好。
alg4 = RandomForestClassifier(random_state=29, n_estimators=180)
alg5 = AdaBoostClassifier(random_state=29)  # AdaBoost分类器
alg6 = KNeighborsClassifier(n_jobs=-1)  # K近邻算法，n_jobs并行处理设置
alg7 = XGBClassifier(random_state=29, n_jobs=-1)  # n_estimators总共迭代的次数，即决策树的个数
alg8 = XGBClassifier(n_estimators=140, random_state=29, n_jobs=-1)  # min_child_weight决定最小叶子节点样本权重和
alg9 = XGBClassifier(n_estimators=140, max_depth=4, min_child_weight=5, random_state=29, n_jobs=-1)
# 列出需要调整的参数范围
parameters1 = {'max_depth': range(1, 10), 'min_samples_split': range(2, 10)}
parameters2 = {"C": range(1, 20), "gamma": [0.05, 0.1, 0.15, 0.2, 0.25]}
parameters3_1 = {'n_estimators': range(10, 200, 10)}
parameters3_2 = {'max_depth': range(1, 10), 'min_samples_split': range(2, 10)}  # 搜索空间太大，分两次调整参数
parameters4 = {'n_estimators': range(10, 200, 10), 'learning_rate': [i / 10.0 for i in range(5, 15)]}
parameters5 = {'n_neighbors': range(2, 10), 'leaf_size': range(10, 80, 20)}
parameters6_1 = {'n_estimators': range(10, 200, 10)}
parameters6_2 = {'max_depth': range(1, 10), 'min_child_weight': range(1, 10)}
parameters6_3 = {'subsample': [i / 10.0 for i in range(1, 10)],
                 'colsample_bytree': [i / 10.0 for i in range(1, 10)]}  # 搜索空间太大，分三次调整参数

algs = [alg1, alg2, alg3, alg4, alg5, alg6, alg7, alg8, alg9]
paras = [parameters1,
         parameters2,
         parameters3_1,
         parameters3_2,
         parameters4,
         parameters5,
         parameters6_1,
         parameters6_2,
         parameters6_3]

clfs = []
for alg, para in zip(algs, paras):
    clfs.append(fit_model(alg, para))


# 保存结果
def save(clf, i):
    pred = clf.predict(test)
    sub = pd.DataFrame({'PassengerId': Id, 'Survived': pred})
    sub.to_csv(r"res/res_tan_{}.csv".format(i), index=False)


i = 1
for clf in clfs:
    save(clf, i)
    i += 1
