# 1. モデルの読み込み
from tensorflow.keras.applications.vgg16 import VGG16
model = VGG16()
print(model.summary())

# 2.画像の読み込み
import numpy as np
from tensorflow.keras.preprocessing import image
filename = "images//fluit2.jpg"
img = image.load_img(filename, target_size=(224, 224))
x_orig = image.img_to_array(img)  # ndarray: (224, 224, 3), float32
x = np.expand_dims(x_orig, axis=0)  # ndarray: (1, 224, 224, 3), float32

# 3. 推論を行い、結果を取得する
from tensorflow.keras.applications.vgg16 import preprocess_input
x_processed = preprocess_input(x)  # ndarray: (1, 224, 224, 3), float32
y_pred = model.predict(x_processed)  # ndarray: (1, 1000), float32

# 4. 推論結果の解釈
from tensorflow.keras.applications.vgg16 import decode_predictions
results = decode_predictions(y_pred, top=5)[0]  # (クラス名, クラス表記, スコア)のリスト
for result in results:
    print(result)
ranking = y_pred[0].argsort()[::-1]
class_idx = ranking[0]

# 5. 入力画像に対する単純な勾配を可視化する
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
class_output = model.output[:, class_idx]  # Tensor / クラススコア
grad_tensor = K.gradients(class_output, model.input)[0]  # Tensor / クラススコアに対する入力の勾配
grad_func = K.function([model.input], [grad_tensor])  # Function / 勾配の値を算出するための関数
gradient = grad_func([x_processed])[0][0]  # ndarray: (224, 224, 3), float32 / 算出された勾配の値


def visualize_mono_map(map_array, base_image=None, output_path=None):
    if map_array.ndim == 3:
        mono_map = np.sum(np.abs(map_array), axis=2)
    else:
        mono_map = map_array

    minimum_value = mono_map.min()
    maximum_value = np.percentile(mono_map, 90)
    normalized_map = (np.minimum(mono_map, maximum_value) - minimum_value) / (maximum_value - minimum_value)

    if base_image is None:
        plt.imshow(normalized_map, cmap='jet')

    else:
        image_norm = (base_image - base_image.min()) / (base_image.max() - base_image.min())
        overlay = np.stack([normalized_map * image_norm[:,:,i] for i in range(3)], axis=2)
        plt.imshow(overlay)

    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path)


def naive_color_mapping(map_array):
    grad_map = np.maximum(map_array, 0.) / map_array.max()  # 正の勾配を算出し正規化 / ndarray: (224, 224, 3) , float32
    plt.imshow(grad_map)
    plt.show()


visualize_mono_map(gradient, base_image=None, output_path="grad_images/naive_grad.png")
visualize_mono_map(gradient, base_image=x_orig, output_path="grad_images/naive_grad_bg.png")


# 6. SmoothGradで勾配を可視化
n_samples = 100  # ランダムノイズを乗せて生成される画像の数
stdev_spread = 0.1  # ライダムノイズの分散のパラメータ（大きいほどランダムノイズを強くする）
stdev = stdev_spread * (np.max(x_processed) - np.min(x_processed))  # 画像の最大値-最小値でランダムノイズの大きさをスケーリング
total_gradient = np.zeros_like(x_processed)  # ndarray: (1, 224, 224, 3), float32 / 勾配の合計値を加算していく行列（0で初期化）
for i in range(n_samples):
    print("SmoothGrad: {}/{}".format(i+1, n_samples))
    x_plus_noise = x_processed \
        + np.random.normal(0, stdev, x_processed.shape)  # ndarray: (1, 224, 224, 3), float32 / xにノイズを付加
    total_gradient += grad_func([x_plus_noise])[0]  # ndarray: (1, 224, 224, 3), float32 / サンプルに対する勾配を算出して合計値に加算

smooth_grad = total_gradient[0] / n_samples  # ndarray: (224, 224, 3), float32 / 勾配の合計値から平均の勾配を算出


visualize_mono_map(smooth_grad, base_image=None, output_path="grad_images/smooth_grad.png")
visualize_mono_map(smooth_grad, base_image=x_orig, output_path="grad_images/smooth_grad_bg.png")


# 6. Guided Backpropで勾配を算出
import tensorflow as tf
from tensorflow.keras.models import load_model
from os import path

model_save_dir = "model"
model_temp_path = path.join(model_save_dir, "temp_orig.h5")
train_save_path = path.join(model_save_dir, "guided_backprop_ckpt")


@tf.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    gate_g = tf.cast(grad > 0, "float32")
    gate_y = tf.cast(op.outputs[0] > 0, "float32")
    return gate_y * gate_g * grad


model.save(model_temp_path)
with tf.Graph().as_default():
    with tf.Session().as_default():
        K.set_learning_phase(0)
        load_model(model_temp_path)
        session = K.get_session()
        tf.train.export_meta_graph()
        saver = tf.train.Saver()
        saver.save(session, train_save_path)

guided_graph = tf.Graph()
with guided_graph.as_default():
    guided_sess = tf.Session(graph=guided_graph)
    with guided_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
        saver = tf.train.import_meta_graph(train_save_path + ".meta")
        saver.restore(guided_sess, train_save_path)

        imported_y = guided_graph.get_tensor_by_name(model.output.name)[0][class_idx]
        imported_x = guided_graph.get_tensor_by_name(model.input.name)

        guided_grads_node = tf.gradients(imported_y, imported_x)

    guided_feed_dict = {imported_x: x_processed}
    sample_gradient = guided_sess.run(guided_grads_node, feed_dict=guided_feed_dict)[0][0]
    visualize_mono_map(sample_gradient, base_image=None, output_path="grad_images/guided_bp.png")
    visualize_mono_map(sample_gradient, base_image=x_orig, output_path="grad_images/guided_bp_bg.png")

    n_samples = 100
    stdev_spread = 0.1
    stdev = stdev_spread * x_processed.max() - x_processed.min()
    total_gradient = np.zeros_like(x_processed)
    for i in range(n_samples):
        print("SmoothGrad of Guided BackProp: {}/{}".format(i+1, n_samples))
        x_plus_noise = x_processed + np.random.normal(0, stdev, x_processed.shape)
        guided_feed_dict = {imported_x: x_plus_noise}
        sample_gradient = guided_sess.run(guided_grads_node, feed_dict=guided_feed_dict)[0]
        total_gradient += sample_gradient

    guided_smooth_grad = total_gradient[0] / n_samples  # 平均の勾配を算出 / ndarray: (224, 224, 3), float32

    visualize_mono_map(guided_smooth_grad, base_image=None, output_path="grad_images/guided_smooth_grad.png")
    visualize_mono_map(guided_smooth_grad, base_image=x_orig, output_path="grad_images/guided_smooth_grad_bg.png")

# 7. Grad CAMによる可視化
import cv2
convout_tensor = model.get_layer("block5_conv3").output  # convolutionの出力/Tensor

grad_tensor = K.gradients(class_output, convout_tensor)[0]  # 勾配/Tensor
grad_func = K.function([model.input], [convout_tensor, grad_tensor])  #勾配を算出する関数
convout_val, grads_val = grad_func([x_processed])
convout_val, grads_val = convout_val[0], grads_val[0]  # array: (14, 14, 512), float32 (両方とも）

weights = np.mean(grads_val, axis=(0,1))  # チャネルの重み/array: (512,), float32
cam = np.dot(convout_val, weights)  # 畳み込みの出力をチャネルで重みづけ/array, (14, 14), float32
cam = np.maximum(cam, 0)
cam = cv2.resize(cam, (224, 224), cv2.INTER_LINEAR)  # 上記をリサイズ

visualize_mono_map(cam, base_image=None, output_path="grad_images/grad_cam.png")
visualize_mono_map(cam, base_image=x_orig, output_path="grad_images/grad_cam_bg.png")

guided_grad_cam = np.stack([cam * smooth_grad[:,:,i] for i in range(3)], axis=2)
