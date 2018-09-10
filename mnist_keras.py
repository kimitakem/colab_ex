import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.data import Dataset

import time

# tf.enable_eager_execution()

# Parameters
batch_size = 128
num_classes = 10
epochs = 10
img_rows, img_cols = 28, 28
num_training = 60000  # 60000 for full
num_test = 10000  # 10000 for full

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
# dataset_train = Dataset.from_tensor_slices((x_train, y_train)).shuffle(num_training, seed=1).batch(batch_size).repeat()
dataset_train = Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size).repeat()
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
#    model.add(Dense(num_classes, activation='softmax'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def print_eager_separater():
    print("\n====Eager Mode====")


def print_keras_separater():
    print("\n====Keras====")


def print_dataset_separater():
    print("\n====Keras_with_tf.data====")


print_eager_separater()
eager_model = create_model()
optimizer = tf.train.AdamOptimizer()
steps_per_epoch = len(x_train) // batch_size

for e in range(epochs):
    for (batch, (images, labels)) in enumerate(dataset_train):

        with tf.GradientTape() as tape:
            logits = eager_model(images, training=True)
            # loss_value = tf.reduce_mean(
            #     tf.nn.softmax_cross_entropy_with_logits_v2(
            #         logits=logits, labels=labels
            #     ))
            # loss_value = tf.losses.softmax_cross_entropy(labels, logits)
            loss_value = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, logits))

            grads = tape.gradient(loss_value, eager_model.variables)
            optimizer.apply_gradients(zip(grads, eager_model.variables))
            print('\rEpoch {}/{}: {}/{} Loss:{}'.format(e + 1,
                                                        epochs,
                                                        batch * batch_size,
                                                        len(x_train),
                                                        loss_value.numpy()), end="")

        if (batch + 1) % steps_per_epoch == 0:
            break

print_dataset_separater()
data_model = create_model()
data_model.compile(loss=keras.losses.categorical_crossentropy,
                   optimizer=optimizer,
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
              optimizer=optimizer,
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
