import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
train_test_split()

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