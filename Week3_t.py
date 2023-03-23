import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data_size=100
perch_length=np.random.randint(80,440,(1,data_size))/10 #(1,100)
perch_weight=perch_length**2-20*perch_length+140+np.random.randn(1,data_size)*50

print(np.shape(perch_length.T)) #(1,100) 한개로 인식  perch_length.T = (100,1)

train_input, test_input, train_target, test_target = train_test_split(perch_length.T, perch_weight.T, random_state=42)
print("학습데이터Shape: ", train_input.shape,"테스트데이터Shape: ", test_input.shape)
train_input = train_input.reshape(-1,1)
test_input = test_input.reshape(-1,1)

knr = KNeighborsRegressor()
knr.n_neighbors = 3
knr.fit(train_input, train_target)

print(knr.score(test_input,test_target))

test_prediction = knr.predict(test_input)
mae = mean_absolute_error(test_target, test_prediction)
print(mae)

print(knr.predict([[50]]))
distances,indexes = knr.kneighbors([[50]])
plt.scatter(train_input,train_target)
plt.scatter(train_input[indexes], train_target[indexes],marker='D')
plt.scatter(50, 1033, marker='^')
plt.show()

print(np.mean(train_target[indexes]))

print(knr.predict([[100]]))


