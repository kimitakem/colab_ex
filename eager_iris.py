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


#############################################
def loss(model, x, y):
    logits = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)


#############################################
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
plt.plot(x, val_accuracy_results, label="eager_val_acc")
plt.plot(x, keras_hist.history['val_acc'], label="keras_val_acc")
plt.title("accuracy")
plt.legend(loc='best')
plt.show()

plt.plot(x, train_loss_results, label="eager_loss")
plt.plot(x, keras_hist.history['loss'], label="keras_loss")
plt.plot(x, val_loss_results, label="eager_val_loss")
plt.plot(x, keras_hist.history['val_loss'], label="keras_loss")
plt.title("loss")
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

import ipdb
ipdb.set_trace()


######### Keras #############

def show_prediction_and_loss(model, x_train, y_train):
    predictions = model(x_train)
    print("logits")
    print(predictions[:5])

    pred_probs = tf.nn.softmax(predictions)
    print("probabilities")
    print(pred_probs[:5])

    pred_labels = tf.argmax(predictions, axis=1)
    print("pred_labels")
    print(pred_labels[:5])

    l_xe_list = keras.losses.categorical_crossentropy(y_train, pred_probs)
    l_xe = tf.reduce_mean(l_xe_list)
    print("Loss test (Calc with Keras): {}".format(l_xe))

    gt_labels = tf.argmax(y_train, axis=1)
    l = tf.losses.sparse_softmax_cross_entropy(labels=gt_labels, logits=predictions)
    print("Loss test: {}".format(l))

    accuracy = tf.reduce_mean(tf.cast(tf.equal(gt_labels, pred_labels), tf.float32))
    print("Accuracy: {}".format(accuracy))



eager_model = get_model_wo_softmax()
optimizer = get_optimizer()

show_prediction_and_loss(eager_model, x_train, y_train)



def show_grads(model, data):
    features, labels = next(iter(data))
    print(labels)
    loss_init, grads = grad(model, features, labels)
    print("Step: {}, Initial Loss: {}".format(global_step.numpy(),
                                              loss_init.numpy()))

    optimizer.apply_gradients(zip(grads, model.variables), global_step)
    print("Step: {},         Loss: {}".format(global_step.numpy(),
                                              loss(model, features, labels).numpy()))


# keep results for plotting
train_loss_results = []
train_accuracy_results = []

for epoch in range(num_epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()
    # Training loop - using batches of 32

    for b, (x, y) in enumerate(train_data):
        print(b, y.shape)
        # Optimize the model
        loss_value, grads = grad(eager_model, x, y)
        optimizer.apply_gradients(zip(grads, eager_model.variables),
                                  global_step)

        # Track progress
        epoch_loss_avg(loss_value)  # add current batch loss
        # compare predicted label to actual label
        epoch_accuracy(tf.argmax(eager_model(x), axis=1, output_type=tf.int32), y)

    # end epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 5 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))
        show_prediction_and_loss(eager_model, x_train, y_train)


show_prediction_and_loss(eager_model, x_train, y_train)


#############################################

print("eager")
show_prediction_and_loss(eager_model, x_train, y_train)
print("keras")
show_prediction_and_loss(keras_model, x_train, y_train)


#############################################
train_dataset_path = "C:\\Users\\kimim\\.keras\\datasets\\iris_training.csv"
print("Local copy of the train dataset file: {}".format(train_dataset_path))

# column order in CSV file
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
feature_names = column_names[:-1]
label_name = column_names[-1]

print("Features: {}".format(feature_names))
print("Label: {}".format(label_name))

class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']


train_original = tf.contrib.data.make_csv_dataset(train_dataset_path,
                                                  batch_size,
                                                  column_names=column_names,
                                                  label_name=label_name,
                                                  num_epochs=1)


def pack_features_vector(features, labels):
    features = tf.stack(list(features.values()), axis=1)
    return features, labels


train_data = train_original.map(pack_features_vector)  # 特徴量をまとめて特徴量ベクトルに変換する
#############################################

def print_all_features(data, n_samples):
    for n, (feature, labels) in enumerate(data):
        print((n, labels, feature))
        if n + 1 >= 10:
            break

