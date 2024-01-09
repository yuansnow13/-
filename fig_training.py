import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import tensorflow as tf
tf.config.run_functions_eagerly(True)
def read_image(paths):
    """
    读取指定路径下的bmp格式图像文件
    """
    os.listdir(paths)
    filelist = []
    for root, dirs, files in os.walk(paths):
        for file in files:
            if os.path.splitext(file)[1] == ".bmp":
                filelist.append(os.path.join(root, file))
    return filelist

def im_array(paths):
    """
    将图像文件转换为灰度图的数组形式
    """
    M = []
    for filename in paths:
        im = Image.open(filename)
        im_L = im.convert("L")
        core = im_L.getdata()
        arr1 = np.array(core, dtype='float32') / 255.0
        list_img = arr1.tolist()
        M.extend(list_img)
    return M

path_1 = 'figs/weiwang/'
path_2 = 'figs/tuowang/'
path_3 = 'figs/ciwang/'

filelist_1 = read_image(path_1)
filelist_2 = read_image(path_2)
filelist_3 = read_image(path_3)
filelist_all = filelist_1 + filelist_2 + filelist_3

M = im_array(filelist_all)

dict_label = {0: '围网', 1: '拖网', 2: '刺网'}
train_images = np.array(M).reshape(len(filelist_all), 224, 224)
label = [0] * len(filelist_1) + [1] * len(filelist_2) + [2] * len(filelist_3)
train_labels = np.array(label)  # 数据标签
train_images = train_images[..., np.newaxis]  # 数据图片
print(train_images.shape)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.AveragePooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
model.summary()  # 显示模型的架构

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
              run_eagerly=True)

model.fit(train_images, train_labels, epochs=1)
#model.save('my_model.h5')  # 保存为h5模型
tf.keras.models.save_model(model, "E:\model")  # 这样是pb模型
print("模型保存成功！")
