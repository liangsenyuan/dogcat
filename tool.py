import random
import numpy as np
from PIL import Image
from keras.saving.save import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model = load_model("model/model1.h5")

#读取随机图片
def read_image():
    folder=r"data/test/"
    file_path = folder + random.choice(os.listdir(folder))
    pil_im = Image.open(file_path,'r')
    return pil_im

def get_predict(pil_im,model):
    # 对图片进行缩放
    pil_im = pil_im.resize((200,200))
    #将格式转为numpy array 格式
    array_im = np.asarray(pil_im)
    array_im = np.expand_dims(array_im,axis=0)
    #对图片进行预测
    result = model.predict(array_im)
    print(result[0][0])
    if result[0][0]>0.5:
       print("预测结果是：狗")
    else:
       print("预测结果是：猫")


def get_Wpredict(pil_im,model):
    # 对图片进行缩放
    pil_im = pil_im.resize((200,200))
    #将格式转为numpy array 格式
    array_im = np.asarray(pil_im)
    array_im = np.expand_dims(array_im,axis=0)
    #对图片进行预测
    result = model.predict(array_im)
    print(result[0][0])
    if result[0][0]>0.5:
        print("预测结果是：狗")
        return "狗"
    else:
        print("预测结果是：猫")
        return "猫"

