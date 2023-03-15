import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

data_input = np.load('data_input.npy')
data_target = np.load('data_target.npy')
kn = KNeighborsClassifier()
#
# print("입력 데이터의 개수 : " + data_input.size) 3600
# print("타겟 데이터의 개수 : "+ data_target.size) 1800

input_arr = np.array(data_input)
target_arr = np.array(data_target)

np.random.seed(1650)

print(input_arr)
print(input_arr.size)
# 입력 데이터 개수 3600

print(target_arr)
print(target_arr.size)
# 타겟 데이터 개수 1800

index = np.arange(1800)
np.random.shuffle(index)
print(index)


train_input = input_arr[index[:500]]
train_target = target_arr[index[:500]]

print("입력 학습 : " )
print(train_input)
print("타겟 학습 : ")
print(train_target)

test_input = input_arr[index[500:]]
test_target = input_arr[index[500:]]

mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)

print(mean,std)
train_scaled = (train_input - mean) / std
new = ([25,300] - mean) / std
kn.fit(train_scaled, train_target)

print(kn.predict([new]))
print(kn.predict([[25,300]]))
distances, indexes = kn.kneighbors([new])

plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')
plt.scatter(train_scaled[indexes,0],train_scaled[indexes,1], marker='D')

print(np.shape(data_input))
print(input_arr)
print(np.shape(data_target))
print(target_arr)

print(np.shape(data_target))
x = np.shape(data_target==1)+np.shape(data_target==2)+np.shape(data_target==3)+np.shape(data_target==4)
print("x 값")
print(x)

a = np.where(data_target==1)
b = np.where(data_target==2)
c = np.where(data_target==3)
d = np.where(data_target==4)

print("a 값 : 1")
print(a)
print("b 값 : 2")
print(b)
print("c 값 : 3")
print(c)
print("d 값 : 4")
print(d)

data1_size = 500
data2_size = 400
data3_size = 400
data4_size = 500


plt.xlabel('length')
plt.ylabel('weight')


plt.show()


