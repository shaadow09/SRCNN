#-*-coding:utf-8-*-
import numpy as np
from load_data import My_data
from Net import SRCNN
import os
import torch
import cv2 as cv
import matplotlib.pyplot as plt
from torchsummary import summary


def psnr(image, noise):
    diff = image - noise
    mse = np.mean(np.square(diff))
    mse = 10 * np.log10(255 * 255 / mse)    
    return mse

def judge(path, test, result):
    for rgb, out in zip(os.listdir(path), os.listdir(test)):
        image = cv.imread(path+'/'+rgb)
        img = cv.imread(test+'/'+out)
        img = cv.resize(img, image.shape[-2:-4:-1])
        result.append(psnr(image, img))
        print(psnr(image, img))

def test(srcnn):
    # 评判多次超分辨率效果
    psnr_list = []
    path = 'recover'
    test = 'test'
    result = []
    judge(path, test, result)
    psnr_list.append(sum(result)/5)

    for i in range(10):
        test_data = My_data('recover')
        test_data.test(srcnn)
        result = []
        judge(path, test, result)
        psnr_list.append(sum(result)/5)
        print(sum(result)/5)
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.ylabel("PSNR(单位:db)")
    plt.xlabel("超分辨率处理次数")
    plt.plot(psnr_list)
    plt.show()


if __name__ == '__main__':
    file = 'srcnn.t7'
    torch.set_printoptions(precision=10, threshold=None, edgeitems=None, linewidth=None, profile=None)
    if os.path.exists(file):
        srcnn = torch.load(file)
    else:
        srcnn = SRCNN(file)
    srcnn = srcnn.cuda()
    summary(srcnn, (3, 33, 33))
    train_data = My_data('train')  # 读取训练数据集  
    srcnn.train(100, train_data.loader) # 训练模型
    
    # test_data = My_data('test')  # 读取测试数据集
    # test_data.test(srcnn)  # 测试数据
    # test(srcnn)

