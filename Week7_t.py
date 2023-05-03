import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_validate, train_test_split
from tensorflow import keras
import tensorflow as tf
(train_input, train_target), (test_input,test_target) = \
    keras.datasets.fashion_mnist.load_data()

fis ,axs = plt.subplots(1,10,figsize=(10,10))
for i in range(10):
    axs[i].imshow(train_input[i], cmap='gray_r')
    axs[i].axis('off')
plt.show()

print([train_target[i] for i in range(10)])

print(np.unique(train_target, return_counts=True))

