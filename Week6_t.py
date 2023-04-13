# 차원 축소 : 데이터의 특징은 유지하면서 벡터의 길이를 축소하는 작업
# PCA : 주성분 분석(핵심요소 분석) : 분산이 큰 방향으로 첫번째 주성분 vector 형성, 분산이 크면 정보량이 많다는 것
# PCA : 공통적인 부분을 많이 가지고 있을 때
# 데이터가 많다는 것 -> 넓게 퍼져 있는 것

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

fruits = np.load('fruits_300.npy')
print(fruits.shape)

print(fruits[0,0, :])
plt.imshow(fruits[0], cmap='gray_r')
plt.show()

apple = fruits[0:100].reshape(-1, 100*100)
pineapple = fruits[100:200].reshape(-1, 100*100)
banana = fruits[200:300].reshape(-1,100*100)


km = KMeans(n_clusters=3, random_state=42)
fruits_2d = fruits.reshape(-1, 100*100)
km.fit(fruits_2d)

# print(apple.shape)


















