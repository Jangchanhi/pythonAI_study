# 커널 사이즈 별로 모아 둔 것이 특성 맵
from tensorflow import keras

# keras.layers.Conv2D(10, kernel_size=(3,3), activation='relu')
# keras.layers.Conv2D(10, kernel_size=(3,3), activation='relu', padding='same')

from tensorflow import keras
from sklearn.model_selection import cross_validate, train_test_split
import matplotlib.pyplot as plt


(train_input, train_target), (test_input,test_target) = keras.datasets.fashion_mnist.load_data()

train_scaled = train_input/255.0
train_scaled = train_scaled.reshape(-1, 28*28)

train_scaled , val_scaled, train_target, val_target = train_test_split(train_scaled, train_target,test_size=0.2, random_state=42)

# 첫 번째 합성곱 층
model = keras.Sequential()

model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu',padding='same',input_shape=(28,28,1)))

model.add(keras.layers.MaxPooling2D(2))

# 두 번째 합성곱 층 + 완전 연결 층
model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100,activation='relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(10,activation='softmax'))

model.summary()

keras.utils.plot_model(model, show_shapes=True)

model.compile(optimizer='adam', loss='sparse_categorial_crossentropy', metrics='accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2,restore_best_weights=True)
history = model.fit(train_scaled, train_target, epochs=20,validation_data=(val_scaled, val_target),callbacks=[checkpoint_cb, early_stopping_cb])

model.evaluate(val_scaled, val_target)

plt.imshow(val_scaled[0].reshape(28, 28), cmap='gray_r')
plt.show()
preds = model.predict(val_scaled[0:1])
print(preds)

test_scaled = test_input.reshape(-1, 28, 28, 1) / 255.0

model.evaluate(test_scaled, test_target)










