# 1.画像の読み込み
import numpy as np
from tensorflow.keras.preprocessing import image
filename = "images//dog2.jpeg"
img = image.load_img(filename, target_size=(224, 224))
x_orig = image.img_to_array(img)  # ndarray: (224, 224, 3), float32
x = np.expand_dims(x_orig, axis=0)  # ndarray: (1, 224, 224, 3), float32

# 2. モデルの読み込み
from tensorflow.keras.applications.vgg16 import VGG16
model = VGG16()
print(model.summary())

# 3. 推論を行い、結果を取得する
from tensorflow.keras.applications.vgg16 import preprocess_input
x_processed = preprocess_input(x)  # ndarray: (1, 224, 224, 3), float32
y_pred = model.predict(x_processed)  # ndarray: (1, 1000), float32

# 4. 推論結果の解釈
from tensorflow.keras.applications.vgg16 import decode_predictions
results = decode_predictions(y_pred, top=5)[0]  # (クラス名, クラス表記, スコア)のリスト
for result in results:
    print(result)
class_idx = np.argmax(y_pred[0])


# 5. 入力画像に対する単純な勾配を可視化する
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
class_output = model.output[:, class_idx]  # class出力 / Tensor
grad_tensor = K.gradients(class_output, model.input)[0]  # class出力に対する入力の勾配 / Tensor
grad_func = K.function([model.input], [grad_tensor])  # 勾配を算出する関数を定義 /  Function
gradient = grad_func([x_processed])[0][0]  # 勾配の値を算出 / ndarray: (224, 224, 3), float32


def naive_color_mapping(grad_array):
    grad_map = np.maximum(grad_array, 0.) / grad_array.max()  # 正の勾配を算出し正規化 / ndarray: (224, 224, 3) , float32
    plt.imshow(grad_map)
    plt.show()


def naive_mono_mapping(grad_array):
    grad_mono_map = np.sum(np.abs(grad_array), axis=2)
    vmax = np.percentile(grad_array, 99)
    vmin = np.min(grad_mono_map)

    plt.imshow(grad_mono_map, cmap='gray', vmin=vmin, vmax=vmax)
    plt.show()


naive_mono_mapping(gradient)


# 6. SmoothGradで勾配を可視化
stdev_spread = 1.0
n_samples = 5  # originally 50
stdev = stdev_spread * (np.max(x_processed) - np.min(x_processed))
total_gradient = np.zeros_like(x_processed)  # 0で初期化
for i in range(n_samples):
    print("SmoothGrad: {}/{}".format(i+1, n_samples))
    x_plus_noise = x_processed \
        + np.random.normal(0, stdev, x_processed.shape)  # xにノイズを付加 / ndarray: (1, 224, 224, 3), float32
    total_gradient += grad_func([x_processed])[0]  # サンプルに対する勾配を算出して合計する / ndarray: (1, 224, 224, 3), float32

smooth_grad = total_gradient[0] / n_samples  # 平均の勾配を算出 / ndarray: (224, 224, 3), float32

naive_mono_mapping(smooth_grad)


import pdb
pdb.set_trace()

# Guided Backprop:
import tensorflow as tf

@tf.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    gate_g = tf.cast(grad > 0, "float32")
    gate_y = tf.cast(op.outputs[0] > 0, "float32")
    return gate_y * gate_g * grad

model_temp_path = "temp_orig.h5"
model.save(model_temp_path)

from tensorflow.keras.models import load_model
import tensorflow as tf

with tf.Graph().as_default():
    with tf.Session().as_default():
        K.set_learning_phase(0)
        load_model(model_temp_path)
        session = K.get_session()
        tf.train.export_meta_graph()
        saver = tf.train.Saver()
        saver.save(session, 'C:\\Users\\kimim\\PycharmProjects\\colab_ex\\guided_backprop_ckpt')

guided_graph = tf.Graph()
with guided_graph.as_default():
    guided_sess = tf.Session(graph=guided_graph)
    with guided_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
        saver = tf.train.import_meta_graph('C:\\Users\\kimim\\PycharmProjects\\colab_ex\\guided_backprop_ckpt.meta')
        saver.restore(guided_sess, 'C:\\Users\\kimim\\PycharmProjects\\colab_ex\\guided_backprop_ckpt')

        imported_y = guided_graph.get_tensor_by_name(model.output.name)[0][class_idx]
        imported_x = guided_graph.get_tensor_by_name(model.input.name)

        guided_grads_node = tf.gradients(imported_y, imported_x)

    guided_feed_dict = {}
    guided_feed_dict[imported_x] = x_processed

    sample_gradient = guided_sess.run(guided_grads_node, feed_dict=guided_feed_dict)[0][0]

    grad_map = np.maximum(sample_gradient, 0.)
    grad_map = grad_map / grad_map.max()

    plt.imshow(grad_map)
    plt.show()



# 可視化3: Grad CAM
conv_output = model.get_layer("block5_conv3").output
grads = K.gradients(class_output, conv_output)[0]
gradient_function = K.function([model.input], [conv_output, grads])

output, grads_val = gradient_function([x_processed])
output, grads_val = output[0], grads_val[0]

weights = np.mean(grads_val, axis=(0,1))
cam = np.dot(output, weights)

import cv2
cam = cv2.resize(cam, (224, 224), cv2.INTER_LINEAR)
cam = np.maximum(cam, 0)
cam = cam / cam.max()
jetcam = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
jetcam = cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB)
jetcam = (np.float32(jetcam) + array_img[:, :, ::-1] / 255 / 2)

# cv2.imshow('array_img', array_img[:, :, ::-1] / 255 / 2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.imshow('cam', jetcam)
cv2.waitKey(0)
cv2.destroyAllWindows()
