# Multi-Layer Perceptron(다층 퍼셉트론, MLP)을 사용하여 덧셈 계산기를 만들어 보는 실습
from keras.models import Sequential
import numpy as np
import random
from tensorflow import keras
from keras.layers import *
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler, normalize, StandardScaler
# from tensorflow.keras.models import load_model
from random import randint
import tensorflow as tf
from keras.optimizers import Adam
# from keras.optimizer_v2.adamax import Adamax
# from keras.optimizer_v2.rmsprop import RMSprop
# from keras.optimizer_v2.gradient_descent import SGD
from sklearn.metrics import r2_score
from functools import partial
# from tensorflow.keras.callbacks import EarlyStopping
from random import randint
from random import randint
# from numpy import sign, abs, log10
from keras.callbacks import EarlyStopping
# from keras.optimizer_v2.adagrad import Adagrad
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
from random import *
import pandas as pd
import matplotlib.pyplot as plt

# 학습 데이터 생성을 위해 해당 범위에 데이터를 소수로 만들고 더한 값을 리턴하는 것과 정수로 만들고 더한 값을 함수로 만든다.
def add_test_data(data_len,data_size):
    x = np.array([np.random.uniform(-data_len, data_len, size=2)
        for _ in range(data_size)])
    y = np.array([[x[i][0] + x[i][1]] for i in range(data_size)])
    return x,y

def add_test_data2(data_len,data_size):
    x = np.array([np.random.uniform(-data_len, data_len, size=2)
        for _ in range(data_size)])
    y = np.array([[x[i][0] + x[i][1]] for i in range(data_size)])
    return x,y







