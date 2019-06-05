# -*- coding: UTF-8 -*-
import http.server  # 修改后
import json
from ocr import OCRNeuralNetwork
import numpy as np
import random



class JSONHandler(http.server.BaseHTTPRequestHandler):   # 修改后
    """处理接收到的POST请求"""
    def do_POST(self):
        response_code = 200
        response = ""
        var_len = int(self.headers.get('Content-Length'))
        content = self.rfile.read(var_len)
        payload = json.loads(content)

        # 如果是训练请求，训练然后保存训练完的神经网络
        if payload.get('train'):
            nn.train(payload['trainArray'])
            nn.save()
        # 如果是预测请求，返回预测值
        elif payload.get('predict'):
            try:
                print(nn.predict(data_matrix[0]))
                response = {"type": "test", "result": str(nn.predict(payload['image']))}
            except:
                response_code = 500
        else:
            response_code = 400

        self.send_response(response_code)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if response:
            self.wfile.write(json.dumps(response).encode())  # 修改后
        return
