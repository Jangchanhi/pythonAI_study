import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

input_data = pd.read_excel('fp.xlsx')
target_data = pd.read_excel('ep.xlsx')
# print(target_data)

z = np.arrage(-5, 5, 0.1)
phi = 1 / (1 + np.exp(-z))
plt.plot(z, phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()











