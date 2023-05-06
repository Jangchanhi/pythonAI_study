import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from scipy.special import softmax



data = pd.read_csv('midterm.csv')

data_input = data[['Weight','Length','Height','Width']].to_numpy()
data_target = data['Species'].to_numpy()

train_input, test_input , train_target, test_target=train_test_split(data_input, data_target, random_state=42)

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

lr = LogisticRegression()
lr.fit(train_scaled,train_target)
decision = lr.decision_function(test_scaled)
print(np.round(decision, decimals=2))

proba = softmax(decision,axis=1)
print(np.round(proba,decimals=3))


#다중분류
lr = LogisticRegression(C=20,max_iter=159)
lr.fit(train_scaled,train_target)

proba = lr.predict_proba(test_scaled)
print(np.round(proba,decimals=3))
print(lr.coef_.shape,lr.intercept_.shape)

print(lr.score(train_scaled,train_target))
print(lr.score(test_scaled,test_target))


