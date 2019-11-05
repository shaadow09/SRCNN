import cv2 as cv
import numpy as np 


def psnr(image, noise):
    diff = image - noise
    mse = np.mean(np.square(diff))
    mse = 10 * np.log10(255 * 255 / mse)    
    return mse

image = cv.imread('low_woman_GT.bmp')
img = cv.resize(cv.imread('woman_GT.bmp'), (image.shape[1], image.shape[0]))
print(psnr(img, image))