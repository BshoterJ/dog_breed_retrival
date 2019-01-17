# -*- coding: utf-8 -*-
"""
Created on Sat Fib 24 18:23:47 2018

@author: Tang Sheyang
"""
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
import cv2
import os
import re
import sys
import operator
import random
import xml.dom.minidom


# 用于测试集的检测
def compute_and_return(img_path):
    img_width, img_height = 64, 64
    num_classes = 900

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    # 搭建模型
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # 加载训练好的权重
    model.load_weights('third_try.h5')

    # 因为需要取倒数第二层的输出为检测结果进行后续比对，故弹出最后一层
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []

    # 声明用于存储检测结果的变量
    list_of_images = []
    compare = {}
    result = {}
    i = 0

    # 读入test_query中选中的图片
    img = cv2.resize(cv2.imread(img_path), (64, 64)).astype(np.float32)
    img /= 255
    img = np.expand_dims(img, axis=0)

    #读出倒数第二层的输出结果
    dense1_orijin_output = model.predict(img)

    reference_path = './test_reference/'

    # 开始遍历reference文件夹里面的图片
    for dir_item in os.listdir(reference_path):
        full_path = os.path.abspath(os.path.join(reference_path, dir_item))
        if dir_item.endswith('.jpg'):
            img = cv2.resize(cv2.imread(full_path), (64, 64)).astype(np.float32)
            img /= 255
            img = np.expand_dims(img, axis=0)
            dense1_output = model.predict(img)
            # 计算倒数第二层向量间的欧氏距离
            dist = np.sqrt(np.sum(np.square(dense1_orijin_output - dense1_output)))
            compare[dir_item] = dist

    # 将存储照片名称和欧式距离的字典按照欧式距离大小从小到大进行排序
    sorted_compare = sorted(compare.items(), key=operator.itemgetter(1))

    for item in range(len(sorted_compare)):
        key, value = sorted_compare[item]
        result[key] = value

    flag = random.randint(18, 22)

    # 从字典里将相近图片的名称提取出来，并组合成地址，添加到列表中
    for key in result.keys():
        list_of_images.append(reference_path + key)
        i = i + 1
        if i == flag:
            i = 0
            break

    # 写入xml文件
    # doc = xml.dom.minidom.Document()
    # root = doc.createElement('Message')
    # root.setAttribute('Version', '1.0')
    # doc.appendChild(root)
    # node_items = doc.createElement('Items')
    # node_items.setAttribute('name', re.findall(r"\d+", img_path)[0])  # 正则表达式仅将图片名中的数字提取出来
    #
    # for i in range(flag):
    #     node_item = doc.createElement('Item')
    #     node_item.setAttribute('image_name', re.findall(r"\d+", list(result.keys())[i])[0])
    #     node_items.appendChild(node_item)
    #
    # root.appendChild(node_items)
    # fp = open('xml_doc' + '/' + 'result.xml', 'w')
    # doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding='utf-8')

    # 返回列表的前15个元素用于显示
    return list_of_images[:15]


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Usage:%s path_name\r\n" % (sys.argv[0]))
    else:
        list_of_images = compute_and_return(sys.argv[1])
