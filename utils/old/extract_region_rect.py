import numpy as np
import cv2 as cv


def rect_crop(img, delta_x, delta_y):
    dimensions = img.shape
    # y
    height = img.shape[0]
    # x
    width = img.shape[1]
    channels = img.shape[2]
    print('Image Dimension    : ', dimensions)
    print('Image Height       : ', height)
    print('Image Width        : ', width)
    print('Number of Channels : ', channels)
    x_top_left = int(width * (float(delta_x) / 100))
    y_top_left = int(height * (float(delta_y) / 100))
    x_bot_right = width - x_top_left
    y_bot_right = height - y_top_left
    cv.rectangle(img, (x_top_left, y_top_left), (x_bot_right, y_bot_right), (0, 255, 0), 3)
    return img



def main():
    img = cv.imread("/Users/dmitrenkoandrey/PycharmProjects/ml_kaggle_competition1/temp/ISIC_0867357.jpg")
    img = rect_crop(img, 20, 20)
    cv.imwrite('/Users/dmitrenkoandrey/PycharmProjects/ml_kaggle_competition1/temp/ISIC_0867357_cropedv2.jpg', img)



if __name__ == "__main__":
    main()
