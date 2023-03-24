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
distances, indexes = knr.kneighbors([[100]])
plt.scatter(train_input, train_target)
plt.scatter(train_input[indexes], train_target[indexes], marker='D')
plt.scatter(100, 1033, marker='^')
plt.show()

lr = LinearRegression()
lr.fit(train_input, train_target)
print(lr.predict([[50]]))
print(lr.coef_, lr.intercept_)
plt.scatter(train_input, train_target)

# plt.plot([15, 50], [15*lr.coef_+lr.intercept_, 50*lr.coef_+lr.intercept_])

plt.scatter(50, 1241.8, marker='^')
plt.show()
print(lr.score(train_input, train_target))
print(lr.score(test_input, test_target))

train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))


print(train_poly.shape, test_poly.shape)


lr = LinearRegression()
lr.fit(train_poly, train_target)

print(lr.predict([[50**2, 50]]))

print(lr.coef_, lr.intercept_)


# 구간별 직선을 그리기 위해 15에서 49까지 정수 배열을 만듭니다
point = np.arange(15, 50)
# 훈련 세트의 산점도를 그립니다
plt.scatter(train_input, train_target)
# 15에서 49까지 2차 방정식 그래프를 그립니다
plt.plot(point, 1.01*point**2 - 21.6*point + 116.05)
# 50cm 농어 데이터
plt.scatter([50], [1574], marker='^')
plt.show()

#%%

print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))

plt.xlabel('length')
plt.ylabel('weight')





