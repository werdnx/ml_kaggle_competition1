import cv2
import tensorflow as tf
import numpy as np
from model_params import target_size_

#cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)



def to_cv_image(tf_image):
    with tf.compat.v1.Session() as sess:
        cv_image = sess.run(fetches=tf_image)
        return cv_image


def to_tf_image(cv_image):
    return tf.convert_to_tensor(cv_image, dtype=tf.float32)


def normalize_image_tf(tf_img):
    cv_image = to_cv_image(tf_img)
    cv_image = cv2.resize(cv_image, (target_size_, target_size_), interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(target_size_, target_size_))
    image[:, :, 0] = clahe.apply(image[:, :, 0])
    return to_tf_image(image)
