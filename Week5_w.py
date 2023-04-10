import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

fruits = np.load('fruits_300.npy')
print(fruits.shape)
print(fruits[0,0,:])
plt.imshow(fruits[0], cmap='gray_r')
plt.show()

apple = fruits[0:100].reshape(-1, 100*100) #
pineapple = fruits[100:200].reshape(-1, 100*100)
banana = fruits[200:300].reshape(-1, 100*100)

print(apple.shape)

plt.hist(np.mean(apple, axis=1), alpha=0.8)
plt.hist(np.mean(pineapple, axis=1), alpha=0.8)
plt.hist(np.mean(banana, axis=1), alpha=0.8)
plt.legend(['apple', 'pineapple', 'banana'])
plt.show()
#
#
# fig, axs = plt.subplots(1, 3, figsize=(20, 5))
# axs[0].bar(range(10000), np.mean(apple, axis=0))
# axs[1].bar(range(10000), np.mean(pineapple, axis=0))
# axs[2].bar(range(10000), np.mean(banana , axis=0))
# plt.show()
#

# 평균 이미지 그리기
apple_mean = np.mean(apple, axis=0).reshape(100, 100)
pineapple_mean = np.mean(pineapple, axis=0).reshape(100,100)
banana_mean = np.mean(banana, axis=0).reshape(100,100)

fig, axs = plt.subplots(1, 3, figsize = (20,5))
axs[0].imshow(apple_mean, cmap='gray_r')
axs[1].imshow(pineapple_mean, cmap='gray_r')
axs[2].imshow(banana_mean, cmap='gray_r')
plt.show()

# 평균과 가까운 사진 고르기
# 비슷한 것을 빼면 값이 작아진다.

# 각 과일과 고르고자 하는 과일 평균 값의 차이
abs_diff = np.abs(fruits - banana_mean)
# 차이를 평균
abs_mean = np.mean(abs_diff, axis=(1, 2))
print(abs_mean.shape)
# 차이가 적은 상위 100개만 뽑음
apple_index = np.argsort(abs_mean)[:100]
fig, axs = plt.subplots(10, 10, figsize=(10, 10))
for i in range(10):
    for j in range(10):
        axs[i, j].imshow(fruits[apple_index[i * 10 + j]], cmap='gray_r')
        axs[i, j].axis('off')

# 고른 과일 평균 값과 가장 유사한 이미지부터 순서대로 출력
plt.show()

# 군집 (Cluster)
# 무엇인지 모르지만(비지도학습) 사전에 정해진 특징들울 기준으로 K개의 군집으로 분류함

# k-평균
# K-Means 알고리즘
# 1. 임의로 K개의 중심점을 구함
# 2. 각 중심점에서 가장 가까운 샘플들을 찾아냄 -> K 개의 군집이 형성
# 3. 각 군집의 값들의 평균을 구해 새로운 K개의 중심점을 구함
# 4. 클러스터 중심의 변화가 일정 값 이하가 될때 까지 2~3을 반복함

km = KMeans(n_clusters=3, random_state=42)
fruits_2d = fruits.reshape(-1, 100 * 100)
km.fit(fruits_2d)

print(km.labels_)
print(np.unique(km.labels_, return_counts=True))


def draw_fruits(arr, ratio=1):
    n = len(arr)
    rows = int(np.ceil(n / 10))
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols, figsize=(cols * ratio, rows * ratio), squeeze=False)

    for i in range(rows):
        for j in range(cols):
            if i * 10 + j < n:
                axs[i, j].imshow(arr[i * 10 + j], cmap='gray_r')
            axs[i, j].axis('off')
    plt.show()


draw_fruits(fruits[km.labels_ == 0])



