import logging
import sys
import os
import numpy as np
from tqdm import *

# init Logger
BPLogger = logging.getLogger('BPLogger')
BPLogger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter
                             ("%(asctime)s - %(name)s - %(levelname)s: %(message)s"))
BPLogger.addHandler(console_handler)
BPLogger.info(
    'Construct DrainLogger success, current working directory: %s'
    % os.getcwd())


class BP:
    # x为输入层神经元个数，y为隐层神经元个数，z输出层神经元个数
    def __init__(self, x, y, z):
        # 学习率l
        self.l_rate = 0.1
        self.x = x
        self.y = y
        self.z = z
        # 隐层阈值
        self.value1 = np.random.randint(-5, 5, (1, y)).astype(np.float64)
        # 输出层阈值
        self.value2 = np.random.randint(-5, 5, (1, z)).astype(np.float64)
        # 输入层与隐层的连接权重
        self.weight1 = np.random.randint(-5, 5, (x, y)).astype(np.float64)
        # 隐层与输出层的连接权重
        self.weight2 = np.random.randint(-5, 5, (y, z)).astype(np.float64)
        BPLogger.info(
            'Init model success with params: l_rate:%d in:%d v:%d out:%d' % (self.l_rate, self.x, self.y, self.z))
        pass

    def sigmoid(self, num):
        return 1 / (1 + np.exp(-num))

    def train(self, input_set, label_set):
        BPLogger.info('Start training ... ...')
        for i in range(len(input_set)):
            input_line = np.mat(input_set[i]).astype(np.float64)
            output_line = np.mat(label_set[i]).astype(np.float64)
            # 隐藏层输入
            input_1 = np.dot(input_line, self.weight1).astype(np.float64)
            # 隐藏层输出
            output_1 = self.sigmoid(input_1 - self.value1).astype(np.float64)
            # 输出层输入
            input_2 = np.dot(output_1, self.weight2).astype(np.float64)
            # 输出层输出
            output_2 = self.sigmoid(input_2 - self.value2).astype(np.float64)

            a = np.multiply(output_2, 1 - output_2)
            g = np.multiply(a, label_set - output_2)
            b = np.dot(g, np.transpose(self.weight2))
            c = np.multiply(output_1, 1 - output_1)
            e = np.multiply(b, c)

            value1_change = -1 * self.l_rate * e
            value2_change = -1 * self.l_rate * g
            weight1_change = self.l_rate * np.dot(np.transpose(input_set), e)
            weight2_change = self.l_rate * np.dot(np.transpose(output_1), g)

            # 更新参数
            self.value1 += value1_change
            self.value2 += value2_change
            self.weight1 += weight1_change
            self.weight2 += weight2_change
        pass
        BPLogger.info('Train finish.')

    def predict(self, input_set):
        BPLogger.info('Start testing ... ...')
        for i in range(len(input_set)):
            input_line = np.mat(input_set[i]).astype(np.float64)
            # 隐藏层输入
            input_1 = np.dot(input_line, self.weight1).astype(np.float64)
            # 隐藏层输出
            output_1 = self.sigmoid(input_1 - self.value1).astype(np.float64)
            # 输出层输入
            input_2 = np.dot(output_1, self.weight2).astype(np.float64)
            # 输出层输出
            output_2 = self.sigmoid(input_2 - self.value2).astype(np.float64)
            print(output_2)


def load_file():
    BPLogger.info('Start load file.')
    fp = open('data.txt')
    # 存放数据
    dataset = []
    # 存放标签
    labelset = []
    for i in fp.readlines():
        a = i.strip().split()
        dataset.append([float(j) for j in a[:len(a) - 2]])
        labelset.append([float(j) for j in a[len(a) - 2:len(a)]])
    return dataset, labelset


if __name__ == '__main__':
    BPLogger.info('Start program.')
    d = np.mat([7, 8, 9])
    e = np.mat([1, 2, 3])
    f = np.c_[d, e]

    print(f.shape)
    # m = BP(3, 3, 2)
    # dataset, labelset = load_file()
    # print(labelset)
    # m.train(dataset, labelset)
    # m.predict(np.mat('59.17 2.95 0.69'))
