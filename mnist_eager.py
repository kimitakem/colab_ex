# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os
import time

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


n_epoch = 2
for e in range(n_epoch):
    for (batch, (images, labels)) in enumerate(train_ds):
        print(batch)
        with tf.GradientTape() as tape:
            logits = model(images, training=True)
            loss_value = loss(logits, labels)
        grads = tape.gradient(loss_value, model.variables)
        optimizer.apply_gradients(
            zip(grads, model.variables)
        )

    import ipdb
    ipdb.set_trace()





