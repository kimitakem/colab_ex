import tensorflow as tf
import numpy as np
from tensorflow.data import Dataset
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.losses import mean_squared_error
from tensorflow.train import GradientDescentOptimizer
import math
tf.enable_eager_execution()

a = 4
b = 10

n_train_samples = 76
batch_size = 10
n_epochs = 10

x_train = 0.01 * np.array(range(n_train_samples)).astype('float32')
y_train = x_train * a + b + np.random.normal(1, 1, n_train_samples)

x_test = 0.1 * np.array(range(10))
y_test = x_test * a + b

train_ds = Dataset.from_tensor_slices((x_train, y_train)).shuffle(n_train_samples).batch(batch_size).repeat()
test_ds = Dataset.from_tensor_slices((x_test,y_test))


for b, item in enumerate(train_ds):
    print(b, item)
    if b + 1 == math.ceil(n_train_samples / batch_size):
        break


def create_model():
    model = Sequential()
    model.add(Dense(4, activation='tanh', input_dim=1))
    model.add(Dense(1))
    return model


eager_model = create_model()
optimizer = GradientDescentOptimizer(0.1)

for e in range(n_epochs):
    for b, (x, y) in enumerate(train_ds):
        with tf.GradientTape() as tape:
            pred = eager_model.predict(x.numpy())
            loss_value = mean_squared_error(pred, y.numpy().reshape(10, 1))
            grads = tape.gradient(loss_value, eager_model.variables)
            optimizer.apply_gradients([grads. eager_model.variables])
            print(loss_value)

    print(e)









