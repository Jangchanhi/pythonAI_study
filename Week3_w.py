# 회귀(Regression) : 기존 데이터에 근거한 예측 2023/03/21 머신러닝 수업
# K-최근접 이웃 회귀

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression # 선형 회귀에 사용되는 함수

# 100개의 길이정보와 무게 정보

# 학습 데이터 준비 : 데이터  Shape 일치 => 학습 데이터 Shape [N,] 으로 되어 있음
# [N,1] Shape 변화가 필요함

data_size=1000
perch_length=np.random.randint(80,440,(1,data_size))/10 #(1,100)
perch_weight=perch_length**2-20*perch_length+110+np.random.randn(1,data_size)*50 #(1,100)
perch_length=perch_length.T
perch_weight=perch_weight.T

train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)

train_input = train_input.reshape(-1,1)
test_input = test_input.reshape(-1,1)

# 회귀모델 훈련
knr = KNeighborsRegressor()
knr.fit(train_input, train_target)

R = knr.score(test_input, test_target)

print("예측이 정확하면 1에 근접함 예측이 Target 평균 수준이라면 0이됨 ")
print(R) # 예측이 정확하면 1에 근접함 예측이 Target 평균 수준이라면 0이됨

test_prediction = knr.predict(test_input)
mae = mean_absolute_error(test_target, test_prediction)
# 절대 값의 차이의 평균을 구함
print("절대 값의 차이의 평균")
print(mae) # 결과 모델 사용 결과 평균 19g 정도의 차이 오차 발생

knr.n_neighbors = 3
knr.fit(train_input, train_target)

print(knr.score(train_input, train_target))
print(knr.score(test_input, test_target))

knr_p = knr.predict([[100]])
print(knr_p)
# 50cm 농어의 이웃을 구한다.
distances,indexes = knr.kneighbors([[50]])

# 훈련 세트의 산점도를 그린다.
plt.scatter(train_input,train_target)
# 훈련 세트 중에서 이웃 샘플만 다시 그린다.
plt.scatter(train_input[indexes], train_target[indexes],marker='D')

# 50cm 농어 데이터
plt.scatter(50, 1050, marker='^')
plt.show()

# 선영 회귀에 사용되는 함수

lr = LinearRegression()
# 선형 회귀 모델 훈련
lr.fit(train_input, train_target)

# 50cm 농어에 대한 예측
print("50cm 농어에 대한 예측")
print(lr.predict([[50]]))
print(lr.coef_, lr.intercept_)

# 훈련 세트의 산점도를 그린다.
plt.scatter(train_input, train_target)
# 15 ~ 50까지 1차 방정식 그래프를 그린다.
# plt.plot([15,50], [15*lr.coef_+lr.intercept_, 50*lr.coef_+lr.intercept_])

# 50cm 농어 데이터
plt.scatter(50, 1241.8, marker='^')
plt.show()
#
# print(lr.score((train_input, train_target)))
print(lr.score(test_input, test_target))



