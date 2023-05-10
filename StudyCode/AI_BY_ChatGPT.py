import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 모델 구성

model = keras.Sequential()

model.add(layers.Dense(64, input_dim=2, activation='relu'))

model.add(layers.Dense(1))

# 모델 컴파일

model.compile(loss='mean_squared_error', optimizer='adam')

# 데이터 준비

x_train = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]

y_train = [0.3, 0.7, 1.1, 1.5, 1.9]

# 모델 학습

model.fit(x_train, y_train, epochs=100, batch_size=2)

# 모델 예측

x_test = [[0.4, 0.5]]

y_pred = model.predict(x_test)

# 결과 출력print(y_pred)