import os
import tensorflow as tf
import io
import PIL.Image as Image

IN_DIR = '/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1/input/new_pos/3/'
OUT_DIR = '/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1/output/malign_candidates/'


# test=validation

def main():
    iter = 0
    tfrecord_files = [(IN_DIR + i, i) for i in os.listdir(IN_DIR)]
    for i, tfrec in enumerate(tfrecord_files):
        record_iterator = tf.python_io.tf_record_iterator(tfrec[0])
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            img_bytes = example.features.feature["image"].bytes_list.value[0]
            image = Image.open(io.BytesIO(img_bytes))
            image.save(OUT_DIR + str(iter) + '.jpg')
            iter = iter + 1
            print('save file ' + OUT_DIR + str(iter) + '.jpg')


if __name__ == "__main__":
    main()
