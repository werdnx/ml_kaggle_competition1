import numpy as np
import cv2 as cv
img = cv.imread('/Users/dmitrenkoandrey/PycharmProjects/ml_kaggle_competition1/temp/ISIC_0867503.jpg')
res = cv.resize(img,None,fx=0.1, fy=0.1, interpolation = cv.INTER_CUBIC)
cv.imwrite('/Users/dmitrenkoandrey/PycharmProjects/ml_kaggle_competition1/temp/ISIC_0867503_zoom.jpg', res)