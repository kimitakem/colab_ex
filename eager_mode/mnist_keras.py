import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras import backend as K
from tensorflow.python.data import Dataset


import time

tf.enable_eager_execution()

# Parameters
batch_size = 32
num_classes = 10
epochs = 5
img_rows, img_cols = 28, 28
num_training = 32  # 60000 for full
num_test = 10  # 10000 for full

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
                     input_shape=input_shape,
                     kernel_initializer='zero', bias_initializer='zero'
                     ))
    model.add(Conv2D(64, (3, 3), activation='relu',
                     kernel_initializer='zero', bias_initializer='zero'
                     ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='zero', bias_initializer='zero'))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer='zero', bias_initializer='zero'))
    return model


def print_eager_separater():
    print("\n====Eager Mode====")


def print_keras_separater():
    print("\n====Keras====")


def print_dataset_separater():
    print("\n====Keras_with_tf.data====")


class Logger(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        print(logs.get('loss'))
        import ipdb
        ipdb.set_trace()

logger = Logger()


print_eager_separater()
eager_model = create_model()
optimizer = tf.train.GradientDescentOptimizer(0.1)
steps_per_epoch = (len(x_train) - 1) // batch_size + 1
validation_steps = (len(x_test) - 1) // batch_size + 1

tfe = tf.contrib.eager

for e in range(epochs):
    for (batch, (images, labels)) in enumerate(dataset_train):

        with tf.GradientTape() as tape:
            logits = eager_model(images, training=True)
            loss_value = tf.losses.softmax_cross_entropy(labels, logits)
        grads = tape.gradient(loss_value, eager_model.variables)
        batch_grads = [4 * batch_size * item for item in grads]
        optimizer.apply_gradients(zip(batch_grads, eager_model.variables))

        print('\rEpoch {}/{}: {}/{} Loss:{}'.format(e + 1,
                                                    epochs,
                                                    batch * batch_size,
                                                    len(x_train),
                                                    loss_value.numpy()),
                                                    end="")

        if (batch + 1) % steps_per_epoch == 0:
            avg_loss = tfe.metrics.Mean('loss', dtype=tf.float32)
            accuracy = tfe.metrics.Accuracy('accuracy', dtype=tf.float32)
            for (val_batch, (val_images, val_labels)) in enumerate(dataset_test):
                predicted_logits = eager_model(val_images, training=False)
                val_loss_value = tf.losses.softmax_cross_entropy(val_labels, predicted_logits)

                avg_loss(val_loss_value)
                accuracy(
                    tf.argmax(val_labels, axis=1, output_type=tf.int64),  # ground truth labels
                    tf.argmax(predicted_logits, axis=1, output_type=tf.int64)  # predicted labels
                )
                if (val_batch + 1) % validation_steps == 0:
                    break

            print("\nValidation Loss: {}, Acc: {}".format(avg_loss.result(), accuracy.result()))
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
                validation_data=(x_test, y_test)
                )
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
