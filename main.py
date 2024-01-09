import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import csv
import datetime
from PIL import Image
import math


class Config:
    trainDir = './data/train'
    testDir = 'E:/test_dataset/test_dataset'
    figDir = './figs'
    savePath = 'result.csv'
    trajectoryGridNum = 100
    trajectoryGridStage = 24
    trajectoryGridStep = 1.0/trajectoryGridStage


config = Config()


def readData(path, model):
    files = os.listdir(path)
    files.sort(key=lambda x: int(x[:-4]))
    # 把所有的数据放在一起，依靠ID来判断
    data = []
    # for i in tqdm(range(len(files)), desc='Reading '+model+' data', leave=True, unit='csv', unit_scale=True):
    for i in tqdm(range(len(files)), desc='Reading '+model+' data', leave=True, unit='csv', unit_scale=True):
        filePath = path + '/' + files[i]
        # 打开文件
        with open(filePath, encoding='utf-8') as file:
            next(file)
            reader = csv.reader(file)
            for row in reader:
                data.append(row)
    dataFrame = pd.DataFrame(data)
    # 判断是训练集还是测试集
    if model == 'train':
        dataFrame.columns = ['ID', 'lat', 'lon',
                             'speed', 'dir', 'time', 'type']  # lat：纬度，lon：经度
    else:
        dataFrame['type'] = 'unknown'
        dataFrame.columns = ['ID', 'lat', 'lon',
                             'speed', 'dir', 'time', 'type']
    # 数据类型转换
    dataFrame['ID'] = dataFrame['ID'].astype(int)
    dataFrame['lat'] = dataFrame['lat'].astype(float)
    dataFrame['lon'] = dataFrame['lon'].astype(float)
    dataFrame['speed'] = dataFrame['speed'].astype(float)
    dataFrame['dir'] = dataFrame['dir'].astype(float)
    dataFrame['time'] = dataFrame['time'].apply(
        lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    return dataFrame


def featureExtraction(dataFrame):
    # 计算差值
    dataFrame['lat_diff'] = dataFrame.groupby('ID')['lat'].diff(1)  # 纬度差分
    dataFrame['lon_diff'] = dataFrame.groupby('ID')['lon'].diff(1)  # 经度差分
    dataFrame['dir_diff'] = dataFrame.groupby('ID')['dir'].diff(1)  # 方向差分

    dataFrame.fillna(0, inplace=True)

    # 提取锚点特征
    dataFrame['anchor'] = dataFrame.apply(lambda x: 1 if x['speed'] == 0 else 0, axis=1)

    # 使用位置信息提取特征
    for id, group_t in dataFrame.groupby('ID'):
        group = pd.DataFrame(group_t)
        group = group.reset_index()
        lat_diff_max = group['lat_diff'].max()
        lat_diff_min = group['lat_diff'].min()
        lat_diff_t = lat_diff_max-lat_diff_min
        lon_diff_max = group['lon_diff'].max()
        lon_diff_min = group['lon_diff'].min()
        lon_diff_t = lon_diff_max - lon_diff_min
        group['position_diff_x'] = group['lat_diff'].apply(lambda x: int(
            round(config.trajectoryGridNum/2*x/lat_diff_t)))
        group['position_diff_y'] = group['lon_diff'].apply(lambda x: int(
            round(config.trajectoryGridNum/2*x/lon_diff_t)))
        rowNum = len(group)
        array = np.ones((config.trajectoryGridNum, config.trajectoryGridNum))
        for index, row in group.iterrows():
            x = int(row['position_diff_x']+config.trajectoryGridNum/2-1)
            y = int(row['position_diff_y']+config.trajectoryGridNum/2-1)
            array[x][y] = array[x][y]-config.trajectoryGridStep
            # array[x][y] = 0.5
            # 锚点位置加重,方向突变位置加重
            if row['anchor'] == 1 or math.fabs(row['dir_diff']) >= 180:
                # array[x][y] = 0.25
                array[x][y] = array[x][y]-4*config.trajectoryGridStep
            # array[x][y] = 0
        if group['type'][0] == '围网':
            path = config.figDir+'/'+'weiwang'
        elif group['type'][0] == '刺网':
            path = config.figDir+'/'+'ciwang'
        elif group['type'][0] == '拖网':
            path = config.figDir+'/'+'tuowang'
        elif group['type'][0] == 'unknown':
            path = config.figDir+'/'+'test'
        path = path + '/' + str(id) + '.bmp'
        print(path)
        outputImg = Image.fromarray(array*255.0)
        outputImg = outputImg.convert('L')
        outputImg.save(path)


if __name__ == "__main__":
    # 获取训练集
    train = readData(config.testDir, 'test')
    # 提取特征
    featureExtraction(train)

