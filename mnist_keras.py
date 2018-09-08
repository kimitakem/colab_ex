import tensorflow.keras as keras
import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.data import Dataset

import time

# Parameters
batch_size = 32
num_classes = 10
epochs = 3
img_rows, img_cols = 28, 28
num_training = 1000
num_test = 100

# Load Original Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocessing
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)[:num_training]
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)[:num_test]
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)[:num_training]
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)[:num_test]
    input_shape = (img_rows, img_cols, 1)

y_train = keras.utils.to_categorical(y_train, num_classes)[:num_training]
y_test = keras.utils.to_categorical(y_test, num_classes)[:num_test]

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# Convert to Dataset
dataset_train = Dataset.from_tensor_slices((x_train, y_train)).shuffle(num_training, seed=1).batch(batch_size).repeat()
dataset_test = Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size).repeat()


def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def print_keras_separater():
    print("\n====Keras====")


def print_dataset_separater():
    print("\n====Keras_with_tf.data====")


print_dataset_separater()
data_model = create_model()
data_model.compile(loss=keras.losses.categorical_crossentropy,
                   optimizer=keras.optimizers.Adadelta(),
                   metrics=['accuracy'])

start = time.time()
data_model.fit(dataset_train.make_one_shot_iterator(),
               epochs=epochs,
               steps_per_epoch=len(x_train) // batch_size,
               validation_data=dataset_test.make_one_shot_iterator(),
               validation_steps=len(x_test) // batch_size,
               verbose=1
               )

data_elapsed_time = time.time() - start
data_score = data_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', data_score[0])
print('Test accuracy:', data_score[1])


print_keras_separater()
keras_model = create_model()
keras_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

start = time.time()
keras_model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(x_test, y_test))
keras_elapsed_time = time.time() - start

keras_score = keras_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', keras_score[0])
print('Test accuracy:', keras_score[1])


print_dataset_separater()
print("Elapsed Time:{0}".format(data_elapsed_time) + "[sec]")
print('Test loss:', data_score[0])
print('Test accuracy:', data_score[1])

print_keras_separater()
print ("keras_elapsed_time:{0}".format(keras_elapsed_time) + "[sec]")
print('Test loss:', keras_score[0])
print('Test accuracy:', keras_score[1])
