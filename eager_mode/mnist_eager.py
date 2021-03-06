# pylint: disable=g-bad-import-order
from absl import app as absl_app
from absl import flags
import tensorflow as tf
# pylint: enable=g-bad-import-order

from official.mnist import dataset as mnist_dataset
from official.mnist import mnist
from official.utils.flags import core as flags_core
from official.utils.misc import model_helpers

# Eager Modeに変更する
tf.enable_eager_execution()
tfe = tf.contrib.eager

train_ds = mnist_dataset.train(".").shuffle(60000).batch(128)
test_ds = mnist_dataset.train(".").batch(128)

import ipdb
ipdb.set_trace()

model = mnist.create_model('channels_last')
optimizer = tf.train.MomentumOptimizer(0.01, 0.01)

train_dir = "train"
test_dir = "test"


def loss(logits, labels):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels
        )
    )


def test(model, dataset):
    avg_loss = tfe.metrics.Mean('loss', dtype=tf.float32)
    for (images, labels) in test_ds:
        logits = model(images, training=False)
        avg_loss(loss(logits, labels))

    print("Test Loss: {}".format(avg_loss.result()))


n_epoch = 10
for e in range(n_epoch):
    for (batch, (images, labels)) in enumerate(train_ds):
        import ipdb
        ipdb.set_trace()
        print('\r{}'.format(batch * 128), end="")
        with tf.GradientTape() as tape:
            logits = model(images, training=True)
            loss_value = loss(logits, labels)
        grads = tape.gradient(loss_value, model.variables)
        optimizer.apply_gradients(zip(grads, model.variables))

        if (batch + 1) % 10 == 0:
            test(model, test_ds)









