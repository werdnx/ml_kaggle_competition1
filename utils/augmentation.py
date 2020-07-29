import cv2
import numpy as np
import albumentations as A
import enum
import os

size = 512

IN_DIR = '/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1/output/malign_candidates/'
OUT_DIR = '/home/werdn/input/jpeg/tsds/train/malignant/'


# OUT_DIR = '/Users/dmitrenkoandrey/PycharmProjects/ml_kaggle_competition1/temp/'


def gaussianBlur(no_hair_img, out_path):
    image = no_hair_img
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), size / 10), -4, 128)
    cv2.imwrite(out_path, image)


def neuron(no_hair_img, out_path):
    image = no_hair_img
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 10), -4, 128)
    cv2.imwrite(out_path, image)


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def circle_crop_img(img, sigmaX=10):
    """
    Create circular crop around image centre
    """

    img = crop_image_from_gray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width, depth = img.shape

    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)
    return img


def circle_crop(no_hair_img, out_path):
    image = no_hair_img
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = circle_crop_img(image)
    cv2.imwrite(out_path, image)


def crop_image1(img, tol=7):
    # img is image data
    # tol  is tolerance
    mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]


def crop_image_from_gray1(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1, img2, img3], axis=-1)
        #         print(img.shape)
        return img


def circle_crop_auto(no_hair_img, out_path):
    image = no_hair_img
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray1(image)
    cv2.imwrite(out_path, image)


fgbg = cv2.createBackgroundSubtractorMOG2()


def backgroundSubtractor(no_hair_img, out_path):
    image = no_hair_img
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = fgbg.apply(image)
    cv2.imwrite(out_path, image)


def imageSegmentation(no_hair_img, out_path):
    image = no_hair_img
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite(out_path, thresh)


def imageSegmentation2(no_hair_img, out_path):
    image = no_hair_img
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    cv2.imwrite(out_path, sure_bg)


def grayscaleImageSegmentation(no_hair_img, out_path):
    image = no_hair_img
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]
    cv2.imwrite(out_path, thresh)


def erosion(no_hair_img, out_path):
    image = no_hair_img
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    kernel = np.ones((5, 5), np.uint8)

    # The first parameter is the original image,
    # kernel is the matrix with which image is
    # convolved and third parameter is the number
    # of iterations, which will determine how much
    # you want to erode/dilate a given image.
    img_erosion = cv2.erode(image, kernel, iterations=1)
    cv2.imwrite(out_path, img_erosion)


def dilation(no_hair_img, out_path):
    image = no_hair_img
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    kernel = np.ones((5, 5), np.uint8)
    img_erosion = cv2.dilate(image, kernel, iterations=1)
    cv2.imwrite(out_path, img_erosion)


def erosion_dilation(no_hair_img, out_path):
    image = no_hair_img
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    kernel = np.ones((5, 5), np.uint8)
    img_erosion = cv2.erode(image, kernel, iterations=1)
    img_erosion = cv2.dilate(img_erosion, kernel, iterations=1)
    cv2.imwrite(out_path, img_erosion)


def canny_edges(no_hair_img, out_path):
    image = no_hair_img
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = circle_crop_img(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

    image = cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(out_path, image)


func_list = [gaussianBlur, neuron, circle_crop, circle_crop_auto,
             imageSegmentation, imageSegmentation2, grayscaleImageSegmentation, erosion, dilation,
             erosion_dilation
             # , canny_edges
             # backgroundSubtractor,
             ]
albus_list = [A.RandomSunFlare(p=0.8), A.GaussNoise(p=0.8), A.CLAHE(p=0.9),
              A.RandomRain(p=1), A.Rotate(p=1, limit=90),
              A.RGBShift(p=1), A.RandomSnow(p=1),
              A.HorizontalFlip(p=1), A.VerticalFlip(p=0.8),
              A.HueSaturationValue(p=1, hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=50)]


def hair_remove(image):
    # convert image to grayScale
    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # kernel for morphologyEx
    kernel = cv2.getStructuringElement(1, (17, 17))

    # apply MORPH_BLACKHAT to grayScale image
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

    # apply thresholding to blackhat
    _, threshold = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # inpaint with original image and threshold image
    final_image = cv2.inpaint(image, threshold, 1, cv2.INPAINT_TELEA)

    return final_image


def aug_iteration(image_file):
    img = cv2.imread(image_file[0], cv2.IMREAD_COLOR)
    img = cv2.resize(img, (size, size))
    no_hair_img = hair_remove(img)
    for aug in func_list:
        out_path = OUT_DIR + aug.__name__ + '_' + image_file[1]
        print ("write file " + out_path)
        aug(no_hair_img, out_path)


def albus_iteration(image_file):
    i_albus = 0
    img = cv2.imread(image_file[0], cv2.IMREAD_COLOR)
    img = cv2.resize(img, (size, size))
    chosen_image = hair_remove(img)
    for albus in albus_list:
        out_path = OUT_DIR + 'albus' + str(i_albus) + '_' + image_file[1]
        img = albus(image=chosen_image)['image']
        cv2.imwrite(out_path, img)
        i_albus = i_albus + 1
        print ("write file " + out_path)


def main():
    test_images = [(IN_DIR + i, i) for i in os.listdir(IN_DIR)]
    for i, image_file in enumerate(test_images):
        aug_iteration(image_file)
        albus_iteration(image_file)


if __name__ == "__main__":
    main()
