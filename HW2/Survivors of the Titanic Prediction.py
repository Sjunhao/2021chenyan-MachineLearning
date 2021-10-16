import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 数据读入
train = pd.read_csv(r'Data/train.csv')
test = pd.read_csv(r'Data/test.csv')

print(train.info())
# 画图
# sns.countplot(x=train['Pclass'],hue=train['Survived'])
# sns.countplot(x=train['Sex'], hue=train['Survived'])
"""
结论：在属性Pclass下，Pclass=1存活几率高，Pclass=3死亡机率更大
    在属性Sex下，man更容易存活
    所以这两个属性与是否存活有较强的相关性
"""

temp_train = train.copy()

temp_train['Age'] = train['Age'].map(lambda x: 'yes' if 0 < x < 100 else 'no')
# sns.countplot(x='Age', hue='Survived', data=temp_train)
# plt.show()
# sns.violinplot(hue='Age', x='Survived', data=temp_train)
# plt.show()

temp_train['Age'] = train['Age'].map(lambda
                                         x: 'child' if x < 12 else 'youth' if x > 30 else 'adult' if x < 60 else 'old' if x < 75 else 'tooold' if x >= 75 else 'null')

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

### get_dummies（）可以做one-hot encoding，离散值大小没意义做
features = pd.get_dummies(features)


### 对测试数据处理
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
    scorer = make_scorer(roc_auc_score)
    grid = GridSearchCV(alg, parameters, scoring=scorer, cv=5)  # 自学GridSearchCV及其他自动调参方法，了解他们的优缺点并在此简单列出
    start = time()
    grid = grid.fit(X, y)
    end = time()
    t = round(end - start, 3)
    print(grid.best_params_)
    print('searching time for {} is {} s'.format(alg.__class__.__name__, t))  # 输出搜索时间
    return grid


# 列出需要使用的算法
alg1 = DecisionTreeClassifier(random_state=29)
alg2 = SVC(probability=True, random_state=29)  # 由于使用roc_auc_score作为评分标准，需将SVC中的probability参数设置为True
alg3 = RandomForestClassifier(random_state=29)
alg4 = RandomForestClassifier(random_state=29, n_estimators=180)
alg5 = AdaBoostClassifier(random_state=29)
alg6 = KNeighborsClassifier(n_jobs=-1)
alg7 = XGBClassifier(random_state=29, n_jobs=-1)
alg8 = XGBClassifier(n_estimators=140, random_state=29, n_jobs=-1)
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


def save(clf, i):
    pred = clf.predict(test)
    sub = pd.DataFrame({'PassengerId': Id, 'Survived': pred})
    sub.to_csv("res_tan_{}.csv".format(i), index=False)


i = 1
for clf in clfs:
    save(clf, i)
    i += 1