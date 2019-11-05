import cv2 as cv
import numpy as np
import os
import math
import torch
import torch.utils.data as Data


class My_data(object):
    def __init__(self, path):
        self.path = path  # 图像数据存储路径 train为训练路径 test为测试或者预测路径
        self.x_size = (33, 33)  # 模型输入端大小
        self.y_size = (21, 21)  # 模型输出端大小
        self.num = []  # 记录图像切片后 各个维度块数量
        self.inputs = []  # 输入数据集
        self.labels = []  # 标签数据集（测试或预测时为输出数据集）
        self.times = 2  # 图片缩放倍数
        self.loader = None  # pytorch数据加载器
        self.batch_size = 16  # 训练批次数据数量
        self.predict = []  # 完整预测图片结果
        self.load_data()

    def clip(self, img):
        """
        将图像切片成多个数据块
        :param img:需要切片的图像
        :return:低分辨率图像
        """
        height, width, depth = img.shape
        num = (math.ceil(height / self.y_size[0]), math.ceil(width / self.y_size[1]))  # 计算切块时 各维度块数
        self.num.append(num)

        img = cv.resize(img, (
            self.y_size[1] * num[1], self.y_size[0] * num[0]))  # 将图片大小固定到切块大小整数
        height, width, depth = img.shape  # 更新图片参数信息

        pad = (int((self.x_size[0] - self.y_size[0]) / 2), int((self.x_size[1] - self.y_size[1]) / 2))  # 计算填充大小

        blur_img = cv.resize(cv.resize(img, (int(width / self.times), int(height / self.times))),
                             (width, height))  # 使用插值缩放图片
        output = blur_img
        blur_img = np.pad(blur_img, (pad, pad, (0, 0)), 'constant')  # 图片边缘填充

        # 图片切片部分
        h = 0
        w = 0
        x_h = 0
        x_w = 0
        while 0 <= h <= height - self.y_size[0]:
            while 0 <= w <= width - self.y_size[1]:
                y = img[h:h + self.y_size[0], w:w + self.y_size[1]]
                x = blur_img[x_h: x_h + self.x_size[0], x_w: x_w + self.x_size[1]]

                self.inputs.append(x)
                self.labels.append(y)
                w += self.y_size[1]
                x_w += self.y_size[1]

            h += self.y_size[0]
            w = 0
            x_h += self.y_size[0]
            x_w = 0
        return output

    def load_data(self):
        """
        读取image下所有数据
        :return: 归一化的输入数据
        """
        files = os.listdir(self.path)
        for file in files:  # 读取目标文件夹下所有图片
            image = cv.imread(self.path + '/' + file)
            cv.imwrite('low_image/low_'+file, self.clip(image))
        print(len(files))
        #  批次训练
        self.inputs = torch.from_numpy(np.array(self.inputs, np.float32), ).permute(
            (0, 3, 1, 2)) / 255  # 数据转换成tensor格式并归一化
        labels = torch.from_numpy(np.array(self.labels, np.float32)).permute((0, 3, 1, 2)) / 255
        torch_dataset = Data.TensorDataset(self.inputs, labels)
        self.loader = Data.DataLoader( # 设置torch的数据读取器
            dataset=torch_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

    def recover(self):
        """
        将类中的label中数据恢复成整张图像
        :return:None
        """
        number = 0
        for height_num, width_num in self.num:  # 表示height width切片块数
            image = np.ones((height_num * self.y_size[0], width_num * self.y_size[1], 3)) * 255
            height, width, depth = image.shape
            h = 0
            w = 0
            while h <= height - self.y_size[0]:
                while w <= width - self.y_size[1]:
                    image[h:h + self.y_size[0], w:w + self.y_size[1]] = self.labels[number] * 255
                    number += 1
                    w += self.y_size[1]
                h += self.y_size[0]
                w = 0
            image = np.where(image < 0, 0, image)
            image = np.where(image > 255, 255, image)
            self.predict.append(image)
        for i, img in enumerate(self.predict):
            cv.imwrite('recover/' + str(i) + '.jpg', img)

    def test(self, srcnn):  # 在srcnn上测试数据
        self.labels = srcnn(
            self.inputs.cuda()).detach().cpu()
        self.labels = self.labels.numpy().transpose((0, 2, 3, 1))
        self.recover()
