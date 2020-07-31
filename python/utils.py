import cv2
import tensorflow as tf
import numpy as np
from model_params import target_size_


def to_cv_image(tf_image):
    with tf.Session() as sess:
        cv_image = tf_image.eval(sess)
        return cv_image


def to_tf_image(cv_image):
    return tf.convert_to_tensor(cv_image, dtype=tf.float32)


def normalize_image_tf(tf_img):
    cv_image = to_cv_image(tf_img)
    image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(target_size_, target_size_))
    image[:, :, 0] = clahe.apply(image[:, :, 0])
    return to_tf_image(image)
