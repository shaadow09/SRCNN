import cv2 as cv
import os 
print(os.listdir())
img = cv.imread('0.bmp')
h, w, d = img.shape
img = cv.resize(img ,(int(w/3), int(h/3)))
img = cv.resize(img , (w, h))
cv.imwrite("low_baby_GT.bmp", img)