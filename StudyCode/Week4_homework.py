import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from scipy.stats import  uniform, randint
# Tree Depth Node수 Gini
wine = pd.read_csv('wine.csv')
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

dt = DecisionTreeClassifier(max_depth=10, random_state=42)

train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

sub_input, val_input, sub_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)

scores = cross_validate(dt, train_input, train_target)
print(scores)

print(np.mean(scores['test_score']))

# 분할기를 사용한 교차 검증
# highper paramiter


# 그리드 서치
params = {'min_impurity_decrease' : [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)

df = gs.best_estimator_
print(dt.score(train_input, train_target))

print(gs.best_params_)
print(gs.cv_results_['mean_test_score'])


rgen = randint(0,10)
rgen.rvs(10)


np.unique(rgen.rvs(1000), return_counts=True)

ugen = uniform(0,1)
ugen.rvs(10)

# 랜덤 서치 값을 넣었는데 값이 계속해서 다르게 나오고 원하느 결과가 나오지 않을 때 사용

