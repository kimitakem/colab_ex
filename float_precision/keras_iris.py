import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation
from tensorflow.python.keras import utils

from sklearn import preprocessing


def build_multilayer_perceptron():
    """多層パーセプトロンモデルを構築"""
    model = Sequential()
    model.add(Dense(16, input_shape=(4, )))
    model.add(Activation('relu'))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    return model


if __name__ == "__main__":
    # Kerasの状態
    print(keras.backend.floatx())


    # Irisデータをロード
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    # データの標準化
    X = preprocessing.scale(X)

    # ラベルをone-hot-encoding形式に変換
    # 0 => [1, 0, 0]
    # 1 => [0, 1, 0]
    # 2 => [0, 0, 1]
    Y = utils.to_categorical(Y)

    # 訓練データとテストデータに分割
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, train_size=0.8)
    print(train_X.shape, test_X.shape, train_Y.shape, test_Y.shape)

    # モデル構築
    model = build_multilayer_perceptron()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # モデル訓練
    model.fit(train_X, train_Y, epochs=50, batch_size=1, verbose=1)

    # モデル評価
    loss, accuracy = model.evaluate(test_X, test_Y, verbose=0)
    weights = model.get_weights()
    print(weights)

    print("Accuracy = {:.2f}".format(accuracy))
