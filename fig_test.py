import os
from PIL import Image
import numpy as np
import tensorflow as tf
import re
import csv
from tqdm import tqdm

# 导入图像数据
model = tf.keras.models.load_model('E:/fishing_boat_identification-master/fishing_boat_identification-master/my_model.h5')
model.summary()  # 查看网络结构

print("模型加载完成！")
dict_label = {0: '围网', 1: '拖网', 2: '刺网'}

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

def im_array(path):
    """
    将图像文件转换为灰度图的数组形式
    """
    im = Image.open(path)
    im_L = im.convert("L")  # 模式L
    Core = im_L.getdata()
    arr1 = np.array(Core, dtype='float32') / 255.0
    list_img = arr1.tolist()
    images = np.array(list_img).reshape(-1, 100, 100, 1)
    return images

test = 'D:\pycharm community edition\pythonProject1/figs/test'  # 测试的图片的路径
filelist = read_image(test)
header = ['渔船ID', 'type']
row = []

with open('result.csv', 'w', encoding='utf8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for file, i in zip(filelist, tqdm(range(len(filelist)), desc='Testing data', leave=True, unit='csv', unit_scale=True)):
        a = []
        a = re.findall("\d", file)  # 正则表达式
        a = int("".join(a))
        img = im_array(file)
        # 预测图像
        predictions_single = model.predict(img, verbose=0)
        row = (a, dict_label[np.argmax(predictions_single)])
        writer.writerow(row)
