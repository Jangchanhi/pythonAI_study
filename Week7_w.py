from tensorflow import keras
from sklearn.model_selection import cross_validate, train_test_split


(train_input, train_target), (test_input,test_target) = \
    keras.datasets.fashion_mnist.load_data()

train_scaled = train_input/255.0
train_scaled = train_scaled.reshape(-1, 28*28)

train_scaled , val_scaled, train_target, val_target = \
    train_test_split(train_scaled, train_target,test_size=0.2, random_state=42)


#dense1 = keras.layers.Dense(100, activation='sigmoid', input_shape=(784,))
#dense2 = keras.layers.Dense(10, activation='softmax')

#model = keras.Sequential([dense1,dense2])

#model.summary()

#model = keras.Sequential([
#    keras.layers.Dense(100, activation='sigmoid', input_shape=(784,),name='hidden'),
#    keras.layers.Dense(10, activation='softmax', name='output')], name='패션 MNIST 모델'
#)

model = keras.Sequential()
model.add(keras.layers.Dense(100, activation='sigmoid',input_shape=(784,)))
model.add(keras.layers.Dense(10,activation='softmax'))

#model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')

#model.fit(train_scaled,train_target, epochs=5)
