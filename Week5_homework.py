import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

input_data = pd.read_excel('fp.xlsx')
target_data = pd.read_excel('ep.xlsx')
# print(target_data)

z = np.arrage(-5, 5, 0.1)
phi = 1 / (1 + np.exp(-z))
plt.plot(z, phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()

# /////////////////////////

wine = pd.read_csv('wine.csv')
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

ss = StandardScaler() # -> StandardScaler 때문에 - 값이 나온다.
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

wine.info()


print("와인 데이터 입니다.")









