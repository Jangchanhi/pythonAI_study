# 로지스틱 회귀와 확률적 경사 하강법
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier # 최근접 이웃의 다중 분류
from sklearn.linear_model import  LogisticRegression
from scipy.special import expit
from scipy.special import softmax

fish = pd.read_csv('https://bit.ly/fish_csv')
fish.head()

# 물고기의 7개 종
print("물고기의 7개 종")
print(pd.unique(fish['Species']))

# 5개 정보 (무게, 길이, 등) -> 7개의 물고기에 대한 확률 정보
# 생선에 대한 5개 정보 -> 7개의 생선 종류 데이터로 mapping
fish_input = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy()
print(fish_input[:5])
fish_target = fish['Species'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)
# 확률 계산 방법 : 샘플을 주변 최근접 K 학습 데이터와 비교
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
# 참고하는 최근접 데이터가 3개 이므로 1/3, 2/3, 3/3 세 가지 확률 결과만 도출됨
kn = KNeighborsClassifier(n_neighbors=3)

# 로지스틱 회귀
# 𝑧 = 𝑎 × 무게 + 𝑏 × 길이 + 𝑐 × 대각선 + 𝑑 × 높이 + 𝑒 × 두께 + 𝑓
#   -> 1차식이 됨

# Sigmoid 함수 0 보다 조금 만 커지면 output이 1에 근접하고,
# 0보다 조금만 작아지면 output이 0에 근접하는 특징을 가지고 있음

kn.fit(train_scaled, train_target)

print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))
print(" 총 데이터의 개수 ", np.shape(train_target)) # 1차원 짜리 데이터

print(kn.classes_)
print(kn.predict(test_scaled[:5]))
proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=4))

# (train_target == 'Bream') 도미만 True로 표시됨
# (train_target == 'Smelt') 방어만 True로 표시됨

bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

print(lr.predict(train_bream_smelt[:5]))

print(lr.predict_proba(train_bream_smelt[:5]))

# fit : a,b,c,d,e,f, 값을 계산해 줌

t_target = train_target[0:10]
print(t_target =='Bream') # 벡터 방식으로 for 문을 사용할 필요 없음
print("Bream 이라는 데이터의 개수 : ",np.sum(t_target=='Bream'))

print(lr.coef_, lr.intercept_)
# 𝑧 = 𝑎 × 무게 + 𝑏 × 길이 + 𝑐 × 대각선 + 𝑑 × 높이 + 𝑒 × 두께 + 𝑓

# z값을 구하기
decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)

# Sigmoid 통과한 결과
print(expit(decisions))

# 로직스틱 회귀(다중 분류)

# c는 규제, max_iter는 반복적인 계산 횟수
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)

print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=3))
print("계수와 Y 절편은? : ",lr.coef_.shape, lr.intercept_.shape)

# 소프트 맥스 함수
#
decisions = lr.decision_function(test_scaled[:5])
print(np.round(decisions, decimals=2))

proba = softmax(decisions, axis=1)
print(np.round(proba, decimals=3))


