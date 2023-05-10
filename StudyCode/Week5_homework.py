import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree

input_data = pd.read_excel('fp.xlsx')
target_data = pd.read_excel('ep.xlsx')
# print(target_data)

z = np.arrage(-5, 5, 0.1)
phi = 1 / (1 + np.exp(-z))
plt.plot(z, phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()

# /////////////////////////

wine = pd.read_csv('wine.csv')
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

ss = StandardScaler() # -> StandardScaler 때문에 - 값이 나온다.
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

wine.info()


print("와인 데이터 입니다.")


# 로지스틱 회귀
# 결과 값이 매우 낮다. 2분의 1인데 90이상이어야 함
# 분류 방법을 고수하기 보다 다른 분류 기법을 활용하는 것이 유리함
lr = LogisticRegression()
lr.fit(train_scaled, train_target)

print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

print(lr.coef_, lr.intercept_)

# 결정 트리 방법 사용
dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled, train_target)
# Tree : 특정 변수를 임계값을 구분하여 오차를 최소화하는 N-중 if문을 구성해줌

print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))

plt.figure(figsize=(10,7))
plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()

# GINI 불순도가 0.5 면 계속 분류를 해야 하고 지니 불순도가 0이 되면 분류가 다 되었다는 순수노드가 되었다는 뜻이다.

# 가지치기 방법
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_scaled, train_target)
# Tree : 특정 변수를 임계값을 구분하여 오차를 최소화하는 N-중 if문을 구성해줌

print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))

plt.figure(figsize=(20, 15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()







