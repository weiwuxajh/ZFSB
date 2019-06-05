# -*- coding: UTF-8 -*-
import http.server  # 修改后
import json
from ocr import OCRNeuralNetwork
import numpy as np
import random

# 服务器端配置
HOST_NAME = 'localhost'
PORT_NUMBER = 9000
# 这个值是通过运行神经网络设计脚本得到的最优值
HIDDEN_NODE_COUNT = 15

# 加载数据集
data_matrix = np.loadtxt(open('data.csv', 'rb'), delimiter=',')
data_labels = np.loadtxt(open('dataLabels.csv', 'rb'))

# 转换成list类型
data_matrix = data_matrix.tolist()
data_labels = data_labels.tolist()

# 数据集一共5000个数据，train_indice存储用来训练的数据的序号
train_indice = list(range(5000))
# 打乱训练顺序
random.shuffle(train_indice)

nn = OCRNeuralNetwork(HIDDEN_NODE_COUNT, data_matrix, data_labels, train_indice);



if __name__ == '__main__':
    server_class = http.server.HTTPServer  # 修改后
    httpd = server_class((HOST_NAME, PORT_NUMBER), JSONHandler)

    try:
        # 启动服务器
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    else:
        print("Unexpected server exception occurred.")
    finally:
        httpd.server_close()
