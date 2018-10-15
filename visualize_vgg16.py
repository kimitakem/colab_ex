from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow.keras.backend as K
import sys
import cv2
import matplotlib.pyplot as plt

# 画像の読み込み
filename = "images//dog2.jpeg"
img = image.load_img(filename, target_size=(224, 224))
x_orig = image.img_to_array(img)
array_img = np.asanyarray(img)
x = np.expand_dims(x_orig, axis=0)
x = x.astype('float32')
preprocessed_input = preprocess_input(x)

# モデルの読み込み
model = VGG16()
print(model.summary())

# 推論を行う
predictions = model.predict(preprocessed_input)
results = decode_predictions(predictions, top=5)[0]
for result in results:
    print(result)

class_idx = np.argmax(predictions[0])
class_output = model.output[:, class_idx]
print("class_output", class_output)

# 可視化1: 勾配
input_grad = K.gradients(class_output, model.input)[0]  # クラス出力に対する入力の勾配
input_grad_func = K.function([model.input], [input_grad])  # 勾配を算出する関数
input_grad_val = input_grad_func([preprocessed_input])[0]  # 入力に対して勾配を算出
grad_map = np.maximum(input_grad_val[0], 0.)  # 勾配が正のところだけを切り出し
grad_map = grad_map / grad_map.max()  # 正規化して[0, 1] にする

#plt.imshow(grad_map)
#plt.show()

# 可視化2: SmoothGrad
stdev_spread = 1.0
nsamples = 5  # originally 50
stdev = stdev_spread * (np.max(preprocessed_input) - np.min(preprocessed_input))
total_gradients = np.zeros_like(preprocessed_input)  # 0で初期化
for i in range(nsamples):
    print(i)
    noise = np.random.normal(0, stdev, preprocessed_input.shape)
    x_value_plus_noise = preprocessed_input + noise

    gradients = input_grad_func([preprocessed_input])[0]
    total_gradients += gradients

smooth_grad = total_gradients[0] / nsamples
image = np.sum(np.abs(smooth_grad), axis=2)

vmax = np.percentile(image, 99)
vmin = np.min(image)

plt.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
plt.show()

# 可視化3: Grad CAM
conv_output = model.get_layer("block5_conv3").output
grads = K.gradients(class_output, conv_output)[0]
gradient_function = K.function([model.input], [conv_output, grads])

output, grads_val = gradient_function([preprocessed_input])
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
