import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0,
34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0,
700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

length = bream_length + smelt_length
weight = bream_weight + smelt_weight

fish_data = [[l,w] for l, w in zip(length,weight)]
fish_target = [1]*35 + [0]*14
#train_input = fish_data[:35]
#train_target = fish_target[:35] #도미

#test_input = fish_data[35:]
#test_target = fish_target[35:] # 빙어
kn = KNeighborsClassifier()

input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

np.random.seed(42)

index = np.arange(49)
np.random.shuffle(index)
print(index)

train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]

test_input = input_arr[index[35:]]
test_target = input_arr[index[35:]]



mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)

print(mean,std)
train_scaled = (train_input - mean) / std
new = ([25,150] - mean) / std
kn.fit(train_scaled, train_target)


print(kn.predict([new]))
print(kn.predict([[25,150]]))


distances, indexes = kn.kneighbors([new])
#plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')

#plt.scatter(25,150,marker='^')
#plt.scatter(train_input[indexes,0], train_input[indexes,1],marker='D')

plt.scatter(train_scaled[indexes,0],train_scaled[indexes,1], marker='D')

#plt.xlim((0,1000))
#plt.scatter(test_input[:,0], test_input[:,1])

plt.xlabel('length')
plt.ylabel('weight')

#print(kn.score(test_input,test_target))
plt.show()