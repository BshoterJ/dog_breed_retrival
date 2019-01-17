# -*- coding: utf-8 -*-
"""
Created on Sat Fib 24 18:23:47 2018

@author: Tang Sheyang
"""
from keras.models import  Model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D, Input, AveragePooling2D
from keras.layers import Activation, Flatten, Dense, merge
from keras import backend as K
import numpy as np
import cv2
import os
import sys
import operator
import random
import json
os.environ["CUDA_VISIBLE_DEVICES"]="1"
def identity_block(input_tensor, kernel_size, filters, stage, block):
    """
    The identity_block is the block that has no conv layer at shortcut
    Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, 1, 1, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, kernel_size, kernel_size,
                      border_mode='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """
    conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, 1, 1, subsample=strides,
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, 1, 1, subsample=strides,
                             name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x


def resnet50_model(img_rows, img_cols, color_type=3, num_classes=None, feature = False):
    """
    Resnet 50 Model for Keras
    Model Schema is based on
    https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py
    ImageNet Pretrained Weights
    https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels.h5
    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color
      num_classes - number of class labels for our classification task
    """

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
        img_input = Input(shape=(img_rows, img_cols, color_type))
    else:
        bn_axis = 1
        img_input = Input(shape=(color_type, img_rows, img_cols))

    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    x_classification_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_classification_fc = Flatten()(x_classification_fc)
    x_classification_fc = Dense(num_classes, activation='softmax', name='fc10')(x_classification_fc)

    # Create another model with our customized softmax
    model = Model(img_input, x_classification_fc)

    # Load resnet50 weights
    model.load_weights('resnet50_new_try.h5')

    if feature:
        x_feature_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
        x_feature_fc = Flatten()(x_feature_fc)
        model = Model(img_input, x_feature_fc)

    return model


# for testing
def compute_and_return(img_path):
    img_width, img_height = 224, 224
    channel = 3
    num_classes = 18

    # if K.image_data_format() == 'channels_first':
    #     input_shape = (3, img_width, img_height)
    # else:
    #     input_shape = (img_width, img_height, 3)

    model_classification = resnet50_model(img_width, img_height, channel, num_classes, False)
    model_feature = resnet50_model(img_width, img_height, channel, num_classes, True)

    # vars for storing results
    list_of_images = []
    list_of_reference_feature = {}
    compare = {}
    result = {}
    i = 0
    
    # load saved reference feature from JSON 
    reference_feature_save = open("reference_feature_save", "r")
    reference_feature_save_js = reference_feature_save.read()
    list_of_reference_feature = json.loads(reference_feature_save_js)
    reference_feature_save.close()
    

    # read images in test_query
    img = cv2.resize(cv2.imread(img_path), (224, 224)).astype(np.float32)
    img /= 255
    img = np.expand_dims(img, axis=0)

    # classification
    dog_type = model_classification.predict(img)
  
    

    # feature
    dog_feature = model_feature.predict(img)
    # print(dog_feature)
    
    reference_path = './retrival/'

    # for all img in ./reference
    #for dir_item in os.listdir(reference_path):
    #    full_path = os.path.abspath(os.path.join(reference_path, dir_item))
    #    if dir_item.endswith('.jpg'):
    #        img = cv2.resize(cv2.imread(full_path), (224, 224)).astype(np.float32)
    #        img /= 255
    #        img = np.expand_dims(img, axis=0)
    #        reference_feature = model_feature.predict(img)
    #        list_of_reference_feature[dir_item] = reference_feature[0].tolist()
    #        # compute distance
    #        dist = np.sqrt(np.sum(np.square(dog_feature - reference_feature)))
    #        compare[dir_item] = dist 
    #reference_feature_save.write(json.dumps(list_of_reference_feature))
    
    for key, value in list_of_reference_feature.items():
        compare[key] = np.sqrt(np.sum(np.square(dog_feature[0] - np.array(value))))
    
        
    # up sort by similarity
    sorted_compare = sorted(compare.items(), key=operator.itemgetter(1))

    for item in range(len(sorted_compare)):
        key, value = sorted_compare[item]
        result[key] = value

    flag = random.randint(18, 22)

    # extract similar images' name from dic to form addresses and insert them into list
    for key in result.keys():
        list_of_images.append(reference_path + key)
        i = i + 1
        if i == flag:
            i = 0
            break

    # xml write
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

    # return 15 images for show
    return list_of_images[:15], dog_type


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Usage:%s path_name\r\n" % (sys.argv[0]))
    else:
        list_of_images, dog_type = compute_and_return(sys.argv[1])
        print(list_of_images)
