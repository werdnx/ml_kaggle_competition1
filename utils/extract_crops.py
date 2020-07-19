import os
import cv2

#IN_DIR = '/home/werdn/input/jpeg/train512_nohair/'
IN_DIR = '/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1/input/jpeg/train/'
OUT_DIR = '/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1/input/jpeg/train_croped/'


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



def crop_image(img):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = find_max_contor(contours)
    x_top_left, y_top_left, x_bot_right, y_bot_right = find_rectangle(cnt)
    crop_img = img[y_top_left:y_top_left+(y_bot_right-y_top_left), x_top_left:x_top_left+(x_bot_right-x_top_left)]
    return crop_img


def main():
    imgs = [(IN_DIR + i, i) for i in os.listdir(IN_DIR)]
    for i, image_file in enumerate(imgs[:50]):
        print('start process file ' + image_file[1])
        img = cv2.imread(image_file[0], cv2.IMREAD_COLOR)
        croped_img = crop_image(img)
        cv2.imwrite(OUT_DIR + image_file[1], croped_img)
        print('process file ' + image_file[1])


if __name__ == "__main__":
    main()