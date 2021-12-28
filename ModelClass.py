import sys
from typing import Tuple, Any

import numpy as np
import yaml
import os
import numpy
from PIL import Image
from keras.utils.vis_utils import plot_model
from keras.models import *
from keras.layers import *
import pickle

from numpy import ndarray

from ImageProcess import ImageProcess


class ModelClass:
    """
    模型调用类
    可使用的成员函数有：
    start_train()——进行模型的训练
    predict()——进行一个图片的预测
    predict_validation()——预测根目录下verifycode.jpg的标签
    predict(image)——预测传入的二值化图片image的标签
    """""

    def __init__(self):
        # 获取yaml配置文件
        config_file = open(r'config/configs.yaml', 'r', encoding='utf-8')
        config_content = config_file.read()
        self.config = yaml.load(config_content, Loader=yaml.FullLoader)
        # 定义标签
        self.charset = ['1', '2', '3', 'b', 'c', 'm', 'n', 'v', 'x', 'z']

    def to_matrix(self, string) -> numpy.ndarray:
        """
        将某个数据的标签名称转化为对应的np矩阵作为训练时的标签值
        :param string: 数据的标签名称
        :return: 对应标签的矩阵
        """""
        length = len(string)
        if length == self.config['dataset']['char_length']:
            # 构建一个 har_length * charset_length 长度的矩阵
            matrix = np.zeros(length * self.config['dataset']['charset_length'])
            for i in range(length):
                # 对应位置1
                matrix[self.charset.index(string[i]) + i * len(self.charset)] = 1
        else:
            # 标签的长度出错时终止程序
            sys.exit("数据集标签出现问题！！！")
        return matrix

    def get_max(self, matrix) -> int:
        """
        获取一个矩阵的最大值位置
        :param matrix: 待检测矩阵
        :return: 最大值的位置
        """""
        max = 0
        max_id = 0
        for i in range(len(matrix)):
            if matrix[i] >= max:
                max = matrix[i]
                max_id = i
        return max_id

    def to_string(self, matrix) -> str:
        matrix1 = matrix[0][0:9]
        matrix2 = matrix[0][10:19]
        matrix3 = matrix[0][20:29]
        matrix4 = matrix[0][30:39]
        char1 = self.charset[self.get_max(matrix1)]
        char2 = self.charset[self.get_max(matrix2)]
        char3 = self.charset[self.get_max(matrix3)]
        char4 = self.charset[self.get_max(matrix4)]
        string = ""
        string = string + char1 + char2 + char3 + char4
        return string

    def init_train_data(self) -> Tuple[ndarray, ndarray]:
        """
        初始化训练集
        :return: x_train—数据集 y_train-标签集
        """""
        train_dir = self.config['dataset']['train_dir']
        train_list = os.listdir(train_dir)
        x_train = []
        y_train = []
        for x in train_list:
            if x.endswith('.jpg'):
                x_train.append(numpy.array(Image.open(train_dir + x)))
                y_train.append(x.rstrip('.jpg'))
        y_train = list(y_train)
        x_train = numpy.array(x_train, dtype=numpy.float32)
        x_train = x_train / 255
        # 通道定义。图像的高为22个像素，宽为62个像素，灰度图像为1层
        x_train = x_train.reshape(x_train.shape[0], self.config['dataset']['height'], self.config['dataset']['width'],
                                  1)
        for x in range(len(y_train)):
            y_train[x] = self.to_matrix(y_train[x])
        y_train = numpy.asarray(y_train)
        return x_train, y_train

    def init_test_data(self) -> Tuple[ndarray, ndarray]:
        """
        初始化验证集
        :return: x_test—数据集 y_test-标签集
        """""
        test_dir = self.config['dataset']['test_dir']
        test_list = os.listdir(test_dir)
        x_test = []
        y_test = []
        for x in test_list:
            if x.endswith('.jpg'):
                x_test.append(numpy.array(Image.open(test_dir + x)))
                y_test.append(x.rstrip('.jpg'))
        y_test = list(y_test)
        x_test = numpy.array(x_test, dtype=numpy.float32)
        x_test = x_test / 255
        x_test = x_test.reshape(x_test.shape[0], self.config['dataset']['height'], self.config['dataset']['width'], 1)
        for x in range(len(y_test)):
            y_test[x] = self.to_matrix(y_test[x])
        y_test = numpy.asarray(y_test)
        return x_test, y_test

    def start_train(self):
        """
        进行模型的训练
        数据集、模型保存位置等变量定义于configs.yaml中
        :return: 无返回值
        """""
        # 获取训练集与验证集
        x_train, y_train = self.init_train_data()
        x_test, y_test = self.init_test_data()
        # 定义输入层（通道数及名称）
        inputs = Input(shape=(self.config['dataset']['height'], self.config['dataset']['width'], 1), name='inputs')
        # 定义第一卷积层，输入为输入层数据
        Layer_1_Convolution = Conv2D(32, (3, 3), name="Layer_1_Convolution")(inputs)
        Layer_1_ReLU = Activation('relu', name='Layer_1_ReLU')(Layer_1_Convolution)
        # 定义第二卷积层，输入为第一激活层数据
        Layer_2_Convolution = Conv2D(32, (3, 3), name='Layer_2_Convolution')(Layer_1_ReLU)
        Layer_2_ReLU = Activation('relu', name='Layer_2_ReLU')(Layer_2_Convolution)
        Layer_2_Pool = MaxPooling2D(pool_size=(2, 2), padding='same', name='Layer_2_Pool')(Layer_2_ReLU)
        # 定义第三卷积层，输入为第二池化层数据
        Layer_3_Convolution = Conv2D(64, (3, 3), name='Layer_3_Convolution')(Layer_2_Pool)
        Layer_3_ReLU = Activation('relu', name='Layer_3_ReLU')(Layer_3_Convolution)
        Layer_3_Pool = MaxPooling2D(pool_size=(2, 2), padding='same', name='Layer_3_Pool')(Layer_3_ReLU)
        # 定义第四卷积层，输入为第三池化层数据
        Layer_4_Convolution = Conv2D(64, (3, 3), name='Layer_4_Convolution')(Layer_3_Pool)
        Layer_4_ReLU = Activation('relu', name='Layer_4_ReLU')(Layer_4_Convolution)
        Layer_4_Pool = MaxPooling2D(pool_size=(2, 2), padding='same', name='Layer_4_Pool')(Layer_4_ReLU)
        # 将数据矩阵折叠成一维数组
        x = Flatten()(Layer_4_Pool)
        # 定义随机丢弃
        x = Dropout(0.25)(x)
        # 4个字符识别的全连接层分别进行10分类
        x = [Dense(10, activation='softmax', name='fullconnected%d' % (i + 1))(x) for i in range(4)]
        # 将四个字符识别结果矩阵合并
        outs = Concatenate()(x)
        # 定义模型的输入与输出
        model = Model(inputs=inputs, outputs=outs)
        model.compile(optimizer=self.config['model']['optimizer'], loss=self.config['model']['loss'],
                      metrics=['accuracy'])
        # 控制台输出模型的摘要
        print(model.summary())
        # 到处模型的结构
        plot_model(model, to_file='asset/model_structure.jpg', show_shapes=True)
        # 开始训练模型
        history = model.fit(x_train,
                            y_train,
                            batch_size=self.config['model']['batch'],
                            epochs=self.config['model']['epochs'],
                            verbose=2,
                            validation_data=(x_test, y_test))
        # 定义模型文件名与记录文件名
        filename_str = '{}new_trained_{}_{}_bs_{}_epochs_{}{}'
        current_model_file = filename_str.format(self.config['model']['model_dir'],
                                                 self.config['model']['optimizer'],
                                                 self.config['model']['loss'],
                                                 self.config['model']['batch'],
                                                 self.config['model']['epochs'],
                                                 self.config['model']['model_format'])
        history_file = filename_str.format(self.config['model']['history_dir'],
                                           self.config['model']['optimizer'],
                                           self.config['model']['loss'],
                                           self.config['model']['batch'],
                                           self.config['model']['epochs'],
                                           self.config['model']['history_format'])
        model.save(current_model_file)
        print('已将模型文件保存到：%s ' % current_model_file)
        print(history.history['accuracy'])
        print(history.history.keys())
        with open(history_file, 'wb') as f:
            pickle.dump(history.history, f)
        print('已将模型记录保存到：%s ' % history_file)

    def predict_validation(self) -> str:
        """
        对根目录下的verifycode.jpg进行预测
        从而判断模型
        :return: 预测的结果
        """""
        filename_str = '{}new_trained_{}_{}_bs_{}_epochs_{}{}'
        model = load_model(filename_str.format(self.config['model']['model_dir'],
                                               self.config['model']['optimizer'],
                                               self.config['model']['loss'],
                                               self.config['model']['batch'],
                                               self.config['model']['epochs'],
                                               self.config['model']['model_format']))
        data = []
        img = Image.open('verifycode.jpg')
        process = ImageProcess()
        img = process.image_binarize(img)
        data.append(numpy.array(img))
        data = numpy.array(data, dtype=numpy.float32)
        data = data / 255
        data = data.reshape(data.shape[0], self.config['dataset']['height'], self.config['dataset']['width'], 1)
        predict = model.predict(data)
        result = self.to_string(predict.tolist())
        return result

    def predict(self, input) -> str:
        """
        预测一个二值化处理后的图片
        返回其预测的结果
        :param input: 二值化处理后的图片 
        :return: 预测的标签结果
        """""
        filename_str = '{}new_trained_{}_{}_bs_{}_epochs_{}{}'
        model = load_model(filename_str.format(self.config['model']['model_dir'],
                                               self.config['model']['optimizer'],
                                               self.config['model']['loss'],
                                               self.config['model']['batch'],
                                               self.config['model']['epochs'],
                                               self.config['model']['model_format']))
        data = [numpy.array(input)]
        data = numpy.array(data, dtype=numpy.float32)
        data = data / 255
        data = data.reshape(data.shape[0], self.config['dataset']['height'], self.config['dataset']['width'], 1)
        predict = model.predict(data)
        result = self.to_string(predict.tolist())
        return result
