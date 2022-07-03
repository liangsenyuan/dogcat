

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.saving.save import load_model

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)




# 创建一个cnn模型
class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def define_model():
    #使用Sequential序列模型

    model = Sequential()
    #卷积层(数量，size，激活函数，padding)
    model.add(Conv2D(32, (3,3), activation="relu",padding="same",input_shape=(200,200,3)))
    #最大池化曾
    model.add(MaxPool2D((2,2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    # 最大池化层（2放缩）
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    # 最大池化层（2放缩）
    model.add(MaxPool2D((2, 2)))
    #flatten层
    model.add(Flatten())
    #全连接神经网络层
    model.add(Dense(128,activation="relu"))
    #sigmoid归一
    model.add(Dense(1, activation="sigmoid"))

    #编译模型
    # 优化训练optimizer优化器
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss="binary_crossentropy" , metrics=['accuracy'])
    return model


def train_model():
    # 实例化模型

    model = define_model()

    # 创建图片生成器
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    # target_size 缩放图片
    train_it = datagen.flow_from_directory("data/train/",
                                           class_mode='binary',
                                           batch_size=64,
                                           target_size=(200, 200))
    test_it = datagen.flow_from_directory('data/test',
        class_mode='binary',
        batch_size=64,
        target_size=(200, 200))
    # 训练模型
    history = LossHistory()
    model.fit(train_it,
              steps_per_epoch=len(train_it),
              epochs=30,
              verbose=1,
              callbacks=history)

# 修改model_apth为你自己保存的模型的位置
    print('缺失率：', history.losses)

    with open('log.txt', 'a', encoding='utf-8') as f:

        f.write(str(history.losses))
    # 保存模型
    model_path = 'model/model3.h5'
    model.save(model_path)

    history1 = model.fit(
        train_it,
        steps_per_epoch=len(train_it),
        epochs = 1,
        validation_data = test_it,
        verbose=1
    )

    return history1

