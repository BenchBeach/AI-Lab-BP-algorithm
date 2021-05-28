import logging
import sys
import os
import numpy as np
from tqdm import *

np.set_printoptions(precision=5, suppress=True)

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


def normalization(datingDatamat):
    max_arr = datingDatamat.max(axis=0)
    min_arr = datingDatamat.min(axis=0)
    ranges = max_arr - min_arr
    norDataSet = np.zeros(datingDatamat.shape).astype(np.float64)
    m = datingDatamat.shape[0]
    norDataSet = datingDatamat - np.tile(min_arr, (m, 1))
    norDataSet = norDataSet / np.tile(ranges, (m, 1))
    return norDataSet, max_arr, min_arr


def sigmoid(num):
    return 1 / (1 + np.exp(-num))


class BP:
    # x为输入层神经元个数，y为隐层神经元个数，z输出层神经元个数
    def __init__(self, x, y, z):
        # 学习率l
        self.l_rate = 0.01
        self.x = x
        self.y = y
        self.z = z
        # 隐层阈值
        self.value1 = np.random.randint(-1, 1, (1, y)).astype(np.float64)
        # 输出层阈值
        self.value2 = np.random.randint(-1, 1, (1, z)).astype(np.float64)
        # 输入层与隐层的连接权重
        self.weight1 = np.random.randint(-1, 1, (x, y)).astype(np.float64)
        # 隐层与输出层的连接权重
        self.weight2 = np.random.randint(-1, 1, (y, z)).astype(np.float64)
        BPLogger.info(
            'Init model success with params: l_rate:%s in:%d v:%d out:%d' % (self.l_rate, self.x, self.y, self.z))
        pass

    def train(self, input_set, label_set):
        # BPLogger.info('Start training ... ...')
        for i in range(len(input_set)):
            input_line = np.mat(input_set[i]).astype(np.float64)
            output_line = np.mat(label_set[i]).astype(np.float64)
            # 隐藏层输入
            input_1 = np.dot(input_line, self.weight1).astype(np.float64)
            # 隐藏层输出
            output_1 = sigmoid(input_1 - self.value1).astype(np.float64)
            # 输出层输入
            input_2 = np.dot(output_1, self.weight2).astype(np.float64)
            # 输出层输出
            output_2 = sigmoid(input_2 - self.value2).astype(np.float64)

            a = np.multiply(output_2, 1 - output_2)
            g = np.multiply(a, output_line - output_2)
            b = np.dot(g, np.transpose(self.weight2))
            c = np.multiply(output_1, 1 - output_1)
            e = np.multiply(b, c)

            value1_change = -1 * self.l_rate * e
            value2_change = -1 * self.l_rate * g
            weight1_change = self.l_rate * np.dot(np.transpose(input_line), e)
            weight2_change = self.l_rate * np.dot(np.transpose(output_1), g)

            # 更新参数
            self.value1 += value1_change
            self.value2 += value2_change
            self.weight1 += weight1_change
            self.weight2 += weight2_change
        pass
        # BPLogger.info('Train finish.')

    def predict(self, input_set):
        BPLogger.info('Start testing ... ...')
        for i in range(len(input_set)):
            input_line = np.mat(input_set[i]).astype(np.float64)
            # 隐藏层输入
            input_1 = np.dot(input_line, self.weight1).astype(np.float64)
            # 隐藏层输出
            output_1 = sigmoid(input_1 - self.value1).astype(np.float64)
            # 输出层输入
            input_2 = np.dot(output_1, self.weight2).astype(np.float64)
            # 输出层输出
            output_2 = sigmoid(input_2 - self.value2).astype(np.float64)
            return output_2


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
    return np.mat(dataset), np.mat(labelset)


def shuffle(dataset, labelset):
    indices = np.arange(dataset.shape[0])
    np.random.shuffle(indices)
    new_dataset = dataset[indices]
    new_labelset = labelset[indices]
    return new_dataset, new_labelset


if __name__ == '__main__':
    BPLogger.info('Start program.')
    # m = np.mat('1 4 8;2 3 8;4 8 4').astype(np.float64)
    # n,x,z = normalization(m)
    # print(n)
    m = BP(3, 3, 2)
    dataset, labelset = load_file()
    x, y, z = normalization(np.mat(dataset))
    h, i, j = normalization(np.mat(labelset))
    for t in range(3000):
        dataset, labelset = shuffle(dataset, labelset)
        m.train(x, h)
    # input for predict
    p = np.mat('59.17 2.95 0.69')
    p = p - z
    p = p / (y - z)
    ans = m.predict(p)
    ans = np.multiply(ans, i - j)
    ans += j
    BPLogger.info('Ans is: ')
    BPLogger.info(ans)
