import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import tensorflow as tf

# 启用立即执行模式
tf.config.run_functions_eagerly(True)
# 接下来编写和执行TensorFlow代码
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

im_wid = 100
path_1 = 'figs/weiwang/'
path_2 = 'figs/tuowang/'
path_3 = 'figs/ciwang/'
filelist_1 = read_image(path_1)
filelist_2 = read_image(path_2)
filelist_3 = read_image(path_3)
filelist_all = filelist_1 + filelist_2 + filelist_3

M = im_array(filelist_all)

dict_label = {0: '围网', 1: '拖网', 2: '刺网'}
train_images = np.array(M).reshape(len(filelist_all), im_wid, im_wid)
label = [0] * len(filelist_1) + [1] * len(filelist_2) + [2] * len(filelist_3)
train_labels = np.array(label)  # 数据标签
train_images = train_images[..., np.newaxis]  # 数据图片
print(train_images.shape)
train_images = np.expand_dims(train_images, axis=3).repeat(3, axis=3)
print(train_images.shape)

model = tf.keras.applications.VGG16(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(im_wid, im_wid, 3),
    pooling=None,
    classes=1000,
    classifier_activation="softmax"
)

for layer in model.layers:
    layer.trainable = False

x = tf.keras.layers.Flatten()(model.output)  # 展平
x = tf.keras.layers.Dense(4096, activation='relu')(x)  # 定义全连接
x = tf.keras.layers.Dropout(0.5)(x)  # Dropout
x = tf.keras.layers.Dense(4096, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
predictions = tf.keras.layers.Dense(3, activation='softmax')(x)  # softmax回归，3分类
head_model = tf.keras.Model(inputs=model.input, outputs=predictions)

head_model.compile(optimizer='adam',
                   loss=tf.keras.losses.sparse_categorical_crossentropy,
                   metrics=['accuracy'],
                   run_eagerly=True)
history = head_model.fit(train_images, train_labels, epochs=10)
head_model.save('my_model.h5')
