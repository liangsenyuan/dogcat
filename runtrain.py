
from cnnmod import define_model, train_model
import tensorflow as tf
#=======
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
#======
model = define_model()
if __name__ == "__main__":
    # 训练模型并保存
    history = train_model()
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    print('训练集准确率：', acc,
          '测试集准确率：', val_acc,
          '训练集缺失率', loss,
          '测试集缺失率', val_loss)
# 修改model_apth为你自己保存的模型的位置

