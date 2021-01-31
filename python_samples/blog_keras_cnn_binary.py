import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.datasets import cifar10
from keras.applications import VGG16
from keras.layers import Input, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator


def get_input():
    # CIFAR10データの読み込み
    (X_train, y_train), (X_test, y_test), = cifar10.load_data()
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    # 0番目のクラスと1番目のクラスのデータを結合
    X_train = np.concatenate([X_train[y_train == 0], X_train[y_train == 1]], axis=0)
    y_train = np.concatenate([y_train[y_train == 0], y_train[y_train == 1]], axis=0)
    X_test = np.concatenate([X_test[y_test == 0], X_test[y_test == 1]], axis=0)
    y_test = np.concatenate([y_test[y_test == 0], y_test[y_test == 1]], axis=0)

    return X_train, y_train, X_test, y_test


def get_datagen():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(
        rescale=1. / 255
    )

    return train_datagen, test_datagen


def define_model(height, width):
    # 入力サイズの定義
    input = Input(shape=(height, width, 3))

    # ベースモデル（VGG16）のインスタンスを生成
    base_model = VGG16(input_shape=(height, width, 3), include_top=False, weights='imagenet', input_tensor=input)
    
    # ベースモデルとtrainable属性をFalseに設定
    for layer in base_model.layers:
        layer.trainable = False

    # ベースモデルの出力を取得
    base_output = base_model.output

    # ベースモデルの出力を一次元化
    base_flatten = Flatten()(base_output)

    # 出力層を追加する
    classifier = Dense(1, activation='sigmoid')(base_flatten)

    # モデルを定義する
    model = Model(inputs=input, outputs=classifier)

    # モデルをコンパイルする
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

    return model


if __name__ == '__main__':
    # 訓練・検証データを作成
    X_train, y_train, X_test, y_test = get_input()

    # KerasのImageDataGeneratorを作成
    train_generator, test_generator = get_datagen()

    # 学習モデルを作成
    model = define_model(height=X_train.shape[1], width=X_train.shape[2])

    # モデル構造を出力する
    model.summary()

    # 訓練を実行 
    history = model.fit_generator(
        train_generator.flow(X_train, y_train, batch_size=32),
        validation_data=test_generator.flow(X_test, y_test, batch_size=32),
        epochs=5)
    
    # Accurayの変遷をプロット
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    # Lossの変遷をプロット
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
