############################################
import tensorflow as tf
tf.enable_eager_execution()
tfe = tf.contrib.eager

print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

#############################################
from sklearn import datasets, preprocessing, model_selection
iris = datasets.load_iris()

all_x = preprocessing.scale(iris.data).astype('float32')  # データのスケーリングと標準化
all_y = iris.target

# 訓練データと検証データに分割（8割（120サンプル）を訓練データに）
train_x, val_x, train_y, val_y = model_selection.train_test_split(all_x, all_y, train_size=0.8)

#############################################
from tensorflow import keras
def get_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(3, activation=tf.nn.softmax)
    ])
    return model


#############################################
def get_optimizer():
    return tf.train.GradientDescentOptimizer(learning_rate=0.01)

batch_size = 32
num_epochs = 500


#############################################
def keras_training():
    model = get_model()
    optimizer = get_optimizer()
    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    hist = model.fit(train_x, train_y, epochs=num_epochs, batch_size=batch_size, validation_data=(val_x, val_y))
    return hist


keras_hist = keras_training()


#############################################
from tensorflow.data import Dataset
train_dataset = Dataset.from_tensor_slices((train_x, train_y)).batch(batch_size)
val_dataset = Dataset.from_tensor_slices((val_x, val_y)).batch(batch_size)


#############################################
def get_model_wo_softmax():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(3)
    ])
    return model


def loss(model, x, y):
    logits = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


#############################################
def eager_training():
    model = get_model_wo_softmax()
    optimizer = get_optimizer()
    global_step = tf.train.get_or_create_global_step()

    # 損失関数の値・精度を残しておくためのリスト
    train_loss_list = []
    train_accuracy_list = []
    val_loss_list = []
    val_accuracy_list = []

    for epoch in range(num_epochs):
        epoch_loss_avg = tfe.metrics.Mean()
        epoch_accuracy = tfe.metrics.Accuracy()

        # 学習のループ
        for b, (x, y) in enumerate(train_dataset):
            # Optimize the model
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.variables),
                                      global_step)

            # Track progress
            epoch_loss_avg(loss_value)  # add current batch loss
            # compare predicted label to actual label
            epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

        train_loss_list.append(epoch_loss_avg.result())
        train_accuracy_list.append(epoch_accuracy.result())

        # 検証のループ
        epoch_test_loss_avg = tfe.metrics.Mean()
        epoch_test_accuracy = tfe.metrics.Accuracy()
        for b, (x, y) in enumerate(val_dataset):
            epoch_test_loss_avg(loss(model, x, y))
            epoch_test_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

        val_loss_list.append(epoch_test_loss_avg.result())
        val_accuracy_list.append(epoch_test_accuracy.result())

    return train_loss_list, train_accuracy_list, val_loss_list, val_accuracy_list


train_loss_results, train_accuracy_results, val_loss_results, val_accuracy_results = eager_training()

######################################
import matplotlib.pyplot as plt
import numpy as np
x = range(num_epochs)

plt.plot(x, train_accuracy_results, label="eager_acc")
plt.plot(x, keras_hist.history['acc'], label="keras_acc")
plt.title("accuracy")
plt.legend(loc='best')
plt.show()

plt.plot(x, val_accuracy_results, label="eager_val_acc")
plt.plot(x, keras_hist.history['val_acc'], label="keras_val_acc")
plt.title("val_accuracy")
plt.legend(loc='best')
plt.show()

plt.plot(x, train_loss_results, label="eager_loss")
plt.plot(x, keras_hist.history['loss'], label="keras_loss")
plt.title("loss")
plt.legend(loc='best')
plt.show()

plt.plot(x, val_loss_results, label="eager_val_loss")
plt.plot(x, keras_hist.history['val_loss'], label="keras_loss")
plt.title("val_loss")
plt.legend(loc='best')
plt.show()


######################################
val_loss_difference = []
loss_difference = []
for i in range(20):
    keras_hist = keras_training()
    train_loss_results, train_accuracy_results, val_loss_results, val_accuracy_results = eager_training()
    loss_difference.append(np.array(train_loss_results) - np.array(keras_hist.history['loss']))
    val_loss_difference.append(np.array(val_loss_results) - np.array(keras_hist.history['val_loss']))


for item in loss_difference:
    plt.plot(item)

plt.title("Loss Difference(Eager - Keras)")
plt.show()

for item in val_loss_difference:
    plt.plot(item)

plt.title("Val Loss Difference(Eager - Keras)")
plt.show()

