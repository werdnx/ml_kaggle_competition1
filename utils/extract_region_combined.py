import numpy as np
import cv2 as cv
import os


# delta_x - percentage delta
def extract_crop(img, delta_x, delta_y):
    result = False
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    max_contour = contours[0]
    max_len = len(contours[0])
    dimensions = img.shape
    # y
    height = img.shape[0]
    # x
    width = img.shape[1]
    channels = img.shape[2]
    x_min_len =  int(width * (float(10) / 100))
    y_min_len = int(height * (float(10) / 100))
    for contour in contours:
        y, x, x_width, y_height = cv.boundingRect(contour)
        y_limit = int(height * (float(delta_y) / 100))
        x_limit = int(width * (float(delta_x) / 100))
        if len(contour) > max_len and y >= y_limit and x >= x_limit and y <= (height - y_limit) and x <= (
                width - x_limit) and x_width > x_min_len and y_height > y_min_len:
            max_len = len(contour)
            max_contour = contour
            result = True
    return result, max_contour


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
    return x_top_left, y_top_left, x_bot_right, y_bot_right



def combined_crop(img):
    is_croped, contour = extract_crop(img, 10, 5)
    if is_croped == True:
        x_top_left, y_top_left, x_bot_right, y_bot_right = find_rectangle(contour)
    else:
        x_top_left, y_top_left, x_bot_right, y_bot_right = rect_crop(img, 20, 20)
    #draw cropped rectangle
    #cv.rectangle(img, (x_top_left, y_top_left), (x_bot_right, y_bot_right), (0, 255, 0), 3)
    #crop image
    crop_img = img[y_top_left:y_top_left + (y_bot_right - y_top_left), x_top_left:x_top_left + (x_bot_right - x_top_left)]
    return crop_img


def hair_remove(image):
    # convert image to grayScale
    grayScale = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    # kernel for morphologyEx
    kernel = cv.getStructuringElement(1, (17, 17))
    # apply MORPH_BLACKHAT to grayScale image
    blackhat = cv.morphologyEx(grayScale, cv.MORPH_BLACKHAT, kernel)
    # apply thresholding to blackhat
    _, threshold = cv.threshold(blackhat, 10, 255, cv.THRESH_BINARY)
    # inpaint with original image and threshold image
    final_image = cv.inpaint(image, threshold, 1, cv.INPAINT_TELEA)
    return final_image



IN_DIR = '/Users/dmitrenkoandrey/PycharmProjects/ml_kaggle_competition1/temp/in/'
OUT_DIR = '/Users/dmitrenkoandrey/PycharmProjects/ml_kaggle_competition1/temp/out/'


def main():
    imgs = [(IN_DIR + i, i) for i in os.listdir(IN_DIR)]
    for i, image_file in enumerate(imgs):
        print('start process file ' + image_file[1])
        img = cv.imread(image_file[0], cv.IMREAD_COLOR)
        #if dimension > 2k resize 2k
        if img.shape[0] > 2000 or img.shape[1] > 2000:
            print('img too big, resize it ' + image_file[1])
            new_height = min(2000, img.shape[0])
            new_width = min(2000, img.shape[1])
            img = cv.resize(img, (new_height, new_width), interpolation=cv.INTER_CUBIC)
        print('remove hair ' + image_file[1])
        img = hair_remove(img)
        print('crop ' + image_file[1])
        croped_img = combined_crop(img)
        #???resize???
        if croped_img.shape[0] > 512 or croped_img.shape[1] > 512:
            new_height = min(512, croped_img.shape[0])
            new_width = min(512, croped_img.shape[1])
            croped_img = cv.resize(croped_img, (new_height, new_width), interpolation=cv.INTER_CUBIC)
        cv.imwrite(OUT_DIR + image_file[1], croped_img)
        print('process file ' + image_file[1])


if __name__ == "__main__":
    main()
