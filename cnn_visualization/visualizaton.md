# 学習済ディープラーニングモデルと注視領域の可視化
## 概要
- tensorflow, kerasで利用できる学習済モデルを使って、クラス判定を行う方法と、その注視領域の可視化についてまとめます。
- kerasを使った可視化としては、[deep-vis-keras](https://github.com/experiencor/deep-viz-keras)というライブラリがgithubで公開されているので、これを参考に注視領域の可視化手法をまとめます。

### 開発環境
- 以下の環境で動作検証をしています。
    - OS: Windows10
    - Tensorflow: v1.11

## Kerasでの学習済モデルの利用
- 可視化の前に、Kerasのapplicationパッケージに含まれる、学習済モデルの利用方法について説明します。

### 1. 学習済モデルの読み込み
- applicationパッケージには学習済モデルのクラスが定義されています。利用できる学習済モデルは以下のサイトから調べられます。VGG16, ResNet50から、軽量モデルのmobilenetまで様々なモデルを利用することができます。
    - https://keras.io/ja/applications/
- ここではVGG16を使うので、`VGG16`クラスを読み込みます。

```python
from tensorflow.keras.applications.vgg16 import VGG16
model = VGG16()
print(model.summary())
```

- このコードは、モデルの構成を出力します。出力結果は以下のようになります。

```
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 224, 224, 3)       0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0         
_________________________________________________________________
fc1 (Dense)                  (None, 4096)              102764544 
_________________________________________________________________
fc2 (Dense)                  (None, 4096)              16781312  
_________________________________________________________________
predictions (Dense)          (None, 1000)              4097000   
=================================================================
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
_________________________________________________________________
```

### 2. 判定対象画像の読み込み
- 次に判定対象とする画像を読み込みます。なんの画像でもよいですが、ここでは、筆者が撮影した以下の画像を使います。
![elephant.jpg](https://qiita-image-store.s3.amazonaws.com/0/78508/cbf2752c-2d30-460c-5207-63918aa3b9b5.jpeg)


```python
import numpy as np
from tensorflow.keras.preprocessing import image
image_name = "elephant"
filename = "images//{}.jpg".format(image_name)  # 画像のパスを指定
img = image.load_img(filename, target_size=(224, 224))
x_orig = image.img_to_array(img)  # ndarray: (224, 224, 3), float32
x = np.expand_dims(x_orig, axis=0)  # ndarray: (1, 224, 224, 3), float32
```

- 上記のコードの通り、画像の読み込みには`load_img`が用意されています。これで画像として読み込んだうえ、`img_to_array`を使ってnumpyの配列に変換しています。
- `load_img`は内部でPILを使って画像を読み込んでいるので、当然、PILやnumpyを直接使って変換してもよいですが、kerasで用意されているメソッドだと、前処理も適用してくれるの便利です。

### 3. 判定を行う

```python
from tensorflow.keras.applications.vgg16 import preprocess_input
x_processed = preprocess_input(x)  # ndarray: (1, 224, 224, 3), float32
y_pred = model.predict(x_processed)  # ndarray: (1, 1000), float32
```

- 画像の配列に対して、`preprocess_input`で前処理を行ったうえで`predict`を使って判定を行います。
- applicationsパッケージにはモデルと同様に前後処理も定義されています。使うモデルによって、処理内容は違うので、そのモデルのパッケージにからimportする必要があります。



### 4. 判定スコアを解釈する
- ここで、出力される`y_pred`は、1000次元のベクトルになります。
- これは、VGG16では、1000クラスの分類になるためで、ベクトルのそれぞれの次元が各クラスに該当する確率を示しています。ただし、どの次元が何の次元かがわからないので、それらを解釈する必要があります。


```python
from tensorflow.keras.applications.vgg16 import decode_predictions
results = decode_predictions(y_pred, top=5)[0]  # (クラス名, クラス表記, スコア)のリスト
for result in results:
    print(result)
ranking = y_pred[0].argsort()[::-1]
class_idx = ranking[0]
```

- このコードでは、全1000クラスのうち、Top 5の確率を持つクラスを算出します。`decode_predictions`を使うことで、各クラスに対応する文字列を取得することができます。
- 出力結果は以下のようになります。この画像では、African_elephantがトップになります。

```
('n02504458', 'African_elephant', 0.97382236)
('n02504013', 'Indian_elephant', 0.025460366)
('n01871265', 'tusker', 0.00071138574)
('n02437312', 'Arabian_camel', 4.3508785e-06)
('n01704323', 'triceratops', 1.0919769e-06)
```

## 注目領域の可視化
- ここからは、学習モデルの注視領域を可視化する方法について説明します。

### 勾配の可視化（単純な方法）
- 注視領域を可視化するための、一番基本的な考え方としては、該当クラスのスコア（確率）に影響を与える入力画素を表示することです。
- 具体的な算出方法としては、クラススコア（出力）の各画素（入力）に対する微分（各画素に対する勾配）を算出し、そこから画像を作ります。

```python
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
class_output = model.output[:, class_idx]  # Tensor / クラススコア
grad_tensor = K.gradients(class_output, model.input)[0]  # Tensor / クラススコアに対する入力の勾配
grad_func = K.function([model.input], [grad_tensor])  # Function / 勾配の値を算出するための関数
gradient = grad_func([x_processed])[0][0]  # ndarray: (224, 224, 3), float32 / 算出された勾配の値
```
- Kerasでの勾配計算は、上記のコードのように記述されます。ポイントは以下の通りです。
    - クラススコアのTensorと入力画像のTensorから、勾配を表すTensorを定義
    - Tensorの値をそれを計算するための関数を定義したのち、具体的な入力値である`x_processed`に対して勾配の値を算出


|  元画像  |  注視領域  | 重ね合わせ  |
| --- | --- | ---  |
| ![elephant.jpg](https://qiita-image-store.s3.amazonaws.com/0/78508/cbf2752c-2d30-460c-5207-63918aa3b9b5.jpeg) |  ![elephant_naive_grad.png](https://qiita-image-store.s3.amazonaws.com/0/78508/07f18ed5-4d31-5e2a-bbd1-e21498787a51.png) | ![elephant_naive_grad_bg.png](https://qiita-image-store.s3.amazonaws.com/0/78508/35eca31e-934b-7698-1e34-11a04e49b6d0.png) |

- 出力される勾配を可視化すると上記のようになります。
- カラー画像だと、勾配はチャネルごとに算出されるため、カラー画像として可視化する方法もありますが、今回は、画素ごとに振幅の平均値を使って可視化することにしました。（以下の手法でも同じ）ソースコードは後述します。
- 単純な方法ではありますが、象の領域がボヤっと抽出されたことがわかります。

### SmoothGradによるノイズ削減
- 勾配の可視化はノイズが多いという課題があるため、ノイズを削減するために提案されたのがSmoothgradと呼ばれる手法です。
- これは、ランダムノイズを与えた複数の入力画像に対して、勾配をそれぞれ計算したうえで、平均する手法のことです
- 単純な方法ですが、ノイズを与えても共通して現れる画素のみを抽出することで、単純な勾配よりも注視領域を安定して可視化できます。
```python
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
```

- 上記のコードでは、ランダムノイズを乗せた100枚の画像を入力画像から算出し、for文の中で1枚ずつ勾配を算出して平均を求めて画像を算出しています

|  元画像  |  注視領域  | 重ね合わせ  |
| --- | --- | ---  |
| ![elephant.jpg](https://qiita-image-store.s3.amazonaws.com/0/78508/cbf2752c-2d30-460c-5207-63918aa3b9b5.jpeg) |  ![elephant_smooth_grad.png](https://qiita-image-store.s3.amazonaws.com/0/78508/139d8923-f3ea-302b-958f-cf0d0d4753d9.png) | ![elephant_smooth_grad_bg.png](https://qiita-image-store.s3.amazonaws.com/0/78508/d67a1927-6ac0-9305-da0f-9ffda31b1d9e.png) |


- 出力結果を可視化すると、上記のようになります。
- 単純勾配では、象の領域全体が赤くなっていたのに対して、象の顔や耳付近がより強い特徴として強調されていることがわかります。

### GuidedPackpropagation
- 単純な勾配を算出する場合、クラススコアに対して正の影響を与える画素と、負の影響を与える画素を両方合わせて計算しています。
- 逆伝搬で勾配を算出する各段階で、負の影響を与えるものを取り除き、正の影響を持つものだけで逆伝搬させていく手法をGuidedBackpropagationと呼ばれています。
- ReLUを使っているアーキテクチャの場合、ReLUを逆伝搬させるときに以下の条件で勾配を0にします。
    - 純伝搬時に負の要素に対する勾配は0になる（通常の勾配計算と同じ）
    - 勾配が負であれば、それの要素の勾配も0にする

```python
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

```

- Kerasでの実装は上記のようになり、ややトリッキーですが、ReLU層を、勾配を算出する際に上記の特性を持った層（GuidedReLU層）に置き換えたうえで、勾配を算出しています。


|  元画像  |  注視領域  | 重ね合わせ  |
| --- | --- | ---  |
| ![elephant.jpg](https://qiita-image-store.s3.amazonaws.com/0/78508/cbf2752c-2d30-460c-5207-63918aa3b9b5.jpeg) |  ![elephant_guided_bp.png](https://qiita-image-store.s3.amazonaws.com/0/78508/f6427f53-f163-3725-9e17-dd5a80b50e4f.png) |  ![elephant_guided_bp_bg.png](https://qiita-image-store.s3.amazonaws.com/0/78508/f5e6e992-5ee2-d997-16f5-0d402051736a.png) |  

- 出力結果を可視化すると、上記のようになり、単純な勾配よりは輪郭を抽出する傾向が強まっているように感じられます。

### GradCAM
- これまでの手法は、各画素の出力結果に対する影響度を求めていますが、画素ごとに影響力を求めるため、情報として細かすぎて、結局どこの領域が重要なのかがわかりづらい場合があります。
- CNNでは入力からの出力に近づいた層ほど、解像度が低く、最終的な判定結果と紐づいた特徴を抽出するとされており、領域の抽出には適していると考えられます。
- したがって、出力に最も近い、最後の畳み込み層の特徴マップを可視化すれば、判定結果と最も紐づく領域が得られそうです。そのためには、多くの次元を持つ特徴マップのうち、出力結果に最も関連する次元を求める必要があります。
- そのために、特徴量マップの各次元の出力結果に対する影響度を算出し、特徴量マップの各次元を重みづけ平均して可視化することにより、最終出力に影響度の高い領域を可視化します。

```python
import cv2
convout_tensor = model.get_layer("block5_conv3").output  # convolutionの出力/Tensor

grad_tensor = K.gradients(class_output, convout_tensor)[0]  # 勾配/Tensor
grad_func = K.function([model.input], [convout_tensor, grad_tensor])  #勾配を算出する関数
convout_val, grads_val = grad_func([x_processed])
convout_val, grads_val = convout_val[0], grads_val[0]  # array: (14, 14, 512), float32 (両方とも）

weights = np.mean(grads_val, axis=(0,1))  # チャネルの重み/array: (512,), float32
grad_cam = np.dot(convout_val, weights)  # 畳み込みの出力をチャネルで重みづけ/array, (14, 14), float32
grad_cam = np.maximum(grad_cam, 0)
grad_cam = cv2.resize(grad_cam, (224, 224), cv2.INTER_LINEAR)  # 上記をリサイズ
```

- 実装は上記のようになります。最終出力(`class_output`)に対する畳み込み層の出力(`counvout_tensor`)に対する勾配を算出するための関数を考慮し、勾配を算出します。
- 次に各、チャネルに対して、重要度を表す`weights`を勾配の大きさを元に作成します。最後に畳み込み層の最終出力（`convout_val`）に対して、重要度をかけて足し合わせることにより、GradCAMの出力とします。

|  元画像  |  注視領域  | 重ね合わせ  |
| --- | --- | ---  |
| ![elephant.jpg](https://qiita-image-store.s3.amazonaws.com/0/78508/cbf2752c-2d30-460c-5207-63918aa3b9b5.jpeg) |  ![elephant_grad_cam.png](https://qiita-image-store.s3.amazonaws.com/0/78508/8a9bb54f-b99f-88f8-47bb-6fe087e7bec1.png) | ![elephant_grad_cam_bg.png](https://qiita-image-store.s3.amazonaws.com/0/78508/e3293e97-36e9-b18d-5c4b-25608e33a5a6.png) | 


- 出力結果は上記のようになります。これまでの手法に比べると、解像度が低く、よりエリアに焦点が上がっていることがわかります。

### 手法の比較
- 注視領域の可視化として最も理解しやすいのはGradCAMです。GradCAMの効果を実感するために、一つの画像の中に複数の種類のオブジェクトがある場合に、可視化を行ってみます。
- この画像は、pineappleが判定結果となりますが、GradCAMによる可視化では、パイナップルの写っている場所に着目して、判定を行ったことが明確にわかります。

|  元画像  |  注視領域（GradCAM）  | 重ね合わせ  |
| --- | --- | --- |
| ![fruit.jpg](https://qiita-image-store.s3.amazonaws.com/0/78508/1ea05df4-45c5-3b9b-ffac-4b6897b73b85.jpeg) | ![fruit_grad_cam.png](https://qiita-image-store.s3.amazonaws.com/0/78508/995c47a8-1cd6-f985-a9f3-749db1bff3ec.png) | ![fruit_grad_cam_bg.png](https://qiita-image-store.s3.amazonaws.com/0/78508/e5123abc-3d0f-d8b9-dad0-4d867d18decc.png) |


- これに対して、GradCAM以外の手法で可視化をすると、以下のようになります。
- 単純な勾配やGuidedBackpropagationでは、画像全体のオブジェクトの中で特徴的なところを抽出してしまい、画像中のどこに注目したのコアが、わからない結果になります。SmoothGradでは少しパイナップルの写っている領域が抽出されており、その部分に着目していることがわかりますが、GradCAMほど顕著に表現されるわけではありませんでした。

|  単純な勾配  |  SmoothGrad  |  GuidedBackpropagation  |
| --- | --- | --- |
![fruit_naive_grad.png](https://qiita-image-store.s3.amazonaws.com/0/78508/699465e5-a995-4d72-0a03-2b5075ed95ac.png) | ![fruit_smooth_grad.png](https://qiita-image-store.s3.amazonaws.com/0/78508/375d1660-b5c6-c794-adca-ad4ad4d66358.png) | ![fruit_guided_bp.png](https://qiita-image-store.s3.amazonaws.com/0/78508/1ab8628e-1212-2946-8399-6288811b722f.png) |



### 可視化のための関数
- 主題ではないので、詳細な説明はしませんが、可視化には以下の関数を使って、ヒートマップ表示、重ね合わせ表示を行いました。
- `map_array`が注視領域の可視化をしたarrayです。重ね合わせについては、`base_image`に元画像を指定することで行うことができます。

```python
def visualize_mono_map(map_array, base_image=None, output_path=None):
    if map_array.ndim == 3:
        mono_map = np.sum(np.abs(map_array), axis=2)  # マップがカラーだった場合はモノクロに変換する
    else:
        mono_map = map_array

    # マップを正規化（上位・下位10%の画素は見やすさのため飽和させる）
    minimum_value = np.percentile(mono_map, 10)
    maximum_value = np.percentile(mono_map, 90)
    normalized_map = (np.minimum(mono_map, maximum_value) - minimum_value) / (maximum_value - minimum_value) 
    normalized_map = np.maximum(normalized_map, 0.)

    if base_image is None:
        plt.imshow(normalized_map, cmap='jet')

    else:
        image_norm = (base_image - base_image.min()) / (base_image.max() - base_image.min())  # 背景画像の正規化
        overlay = np.stack([normalized_map * image_norm[:,:,i] for i in range(3)], axis=2)
        plt.imshow(overlay)

    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path)
```

### まとめ
- 以上、学習済ディープラーニングモデルでの推論方法と、その注視領域の可視化について整理をしました。

### 参考
- [ディープラーニングの判断根拠を理解する手法](https://qiita.com/icoxfog417/items/8689f943fd1225e24358)
- [CNNの可視化に関して簡単に紹介しているページをざっくり翻訳した](https://urusulambda.wordpress.com/2018/05/13/cnn%E3%81%AE%E5%8F%AF%E8%A6%96%E5%8C%96%E3%81%AB%E9%96%A2%E3%81%97%E3%81%A6%E7%B0%A1%E5%8D%98%E3%81%AB%E7%B4%B9%E4%BB%8B%E3%81%97%E3%81%A6%E3%81%84%E3%82%8B%E3%83%9A%E3%83%BC%E3%82%B8%E3%82%92/)
- [deep-vis-keras](https://github.com/experiencor/deep-viz-keras)（参考にしたソースコード）




