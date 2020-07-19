import numpy as np
import cv2 as cv


def find_max_contor(p_contours):
    max_contour = p_contours[0]
    max_len = len(p_contours[0])
    for contour in p_contours:
        if len(contour) > max_len:
            max_len = len(contour)
            max_contour = contour
    return max_contour


def find_rectangle(contour):
    delta = 20
    x_top_left = 50000
    y_top_left = 50000
    x_bot_right = 0
    y_bot_right = 0
    for xy in contour:
        if x_top_left > xy[0][0]:
            x_top_left = xy[0][0]
        if x_bot_right < xy[0][0]:
            x_bot_right = xy[0][0]
        if y_top_left > xy[0][1]:
            y_top_left = xy[0][1]
        if y_bot_right < xy[0][1]:
            y_bot_right = xy[0][1]

    return x_top_left - delta, y_top_left - delta, x_bot_right + delta, y_bot_right + delta

img = cv.imread('/Users/dmitrenkoandrey/PycharmProjects/ml_kaggle_competition1/temp/ISIC_0052212.jpg')
imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# cv.drawContours(img, contours, -1, (0,255,0), 3)
# cv.drawContours(img, contours, 70, (0,255,0), 3)
cnt = find_max_contor(contours)

x_top_left, y_top_left, x_bot_right, y_bot_right = find_rectangle(cnt)
crop_img = img[y_top_left:y_top_left+(y_bot_right-y_top_left), x_top_left:x_top_left+(x_bot_right-x_top_left)]


#cv.drawContours(img, [cnt], 0, (0, 255, 0), 3)
#cv.rectangle(img,(x_top_left,y_top_left),(x_bot_right,y_bot_right),(0,255,0),3)
#cv.imwrite('/Users/dmitrenkoandrey/PycharmProjects/ml_kaggle_competition1/temp/ISIC_0052212_1.jpg', img)
cv.imwrite('/Users/dmitrenkoandrey/PycharmProjects/ml_kaggle_competition1/temp/ISIC_0052212_1_croped.jpg', crop_img)


