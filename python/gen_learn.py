from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from efficientnet.keras import EfficientNetB3
from keras import callbacks
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import cv2
import pandas as pd
import time

iteration = '5'

def run():
    train_dir = '/input/jpeg/train512_nohair'
    train_df = pd.read_csv('/input/train.csv')
    train_df['target'] = train_df['target'].astype('str')
    train_df['image_name'] = train_df['image_name'].astype(str)

    pos_df0 = train_df[train_df['target'] == '1']
    pos_df0['image_name'] = pos_df0['image_name'] + '_0.jpg'
    pos_df1 = train_df[train_df['target'] == '1']
    pos_df1['image_name'] = pos_df1['image_name'] + '_1.jpg'
    pos_df2 = train_df[train_df['target'] == '1']
    pos_df2['image_name'] = pos_df2['image_name'] + '_2.jpg'
    pos_df3 = train_df[train_df['target'] == '1']
    pos_df3['image_name'] = pos_df3['image_name'] + '_3.jpg'

    pos_df4 = train_df[train_df['target'] == '1']
    pos_df4['image_name'] = pos_df4['image_name'] + '_4.jpg'
    pos_df5 = train_df[train_df['target'] == '1']
    pos_df5['image_name'] = pos_df5['image_name'] + '_5.jpg'
    pos_df6 = train_df[train_df['target'] == '1']
    pos_df6['image_name'] = pos_df6['image_name'] + '_6.jpg'
    pos_df7 = train_df[train_df['target'] == '1']
    pos_df7['image_name'] = pos_df7['image_name'] + '_7.jpg'

    train_df['image_name'] = train_df['image_name'] + '.jpg'

    frames = [pos_df1, train_df, pos_df0, pos_df2, pos_df3, pos_df4, pos_df5, pos_df6, pos_df7]

    train_df = pd.concat(frames)
    train_df.head()
    print("len of target = 1: " + str(len(train_df[train_df["target"] == '1'])))
    time.sleep(60)

    batch_size_ = batch_size = 30
    #size of images
    target_size_ = 128
    steps_per_epoch_ = len(train_df)/batch_size
    validation_steps_ = steps_per_epoch_ / 5
    train_datagen = ImageDataGenerator(
        rescale=1 / 255,
        validation_split=0.10,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.2, 1.0],
        zoom_range=[0.5, 1.0],
        shear_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=train_dir,
        x_col="image_name",
        y_col="target",
        target_size=(target_size_, target_size_),
        subset="training",
        batch_size = batch_size_,
        shuffle=True,
        class_mode="binary"
    )
    val_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=train_dir,
        x_col="image_name",
        y_col="target",
        target_size=(target_size_, target_size_),
        subset="validation",
        batch_size = batch_size_,
        shuffle=True,
        class_mode="binary"
    )
    efficient_net = EfficientNetB3(
        weights='imagenet',
        input_shape=(target_size_, target_size_, 3),
        include_top=False,
        pooling='max'
    )

    model = Sequential()
    model.add(efficient_net)
    model.add(Dense(units=120, activation='relu'))
    model.add(Dense(units=120, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.summary()
    model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    #steps_per_epoch = train_size/ batches
    history = model.fit(
        train_generator,
        epochs=36,
        steps_per_epoch=steps_per_epoch_,
        validation_data=val_generator,
        validation_steps=validation_steps_,
        verbose=1,
    )
    model.save('/output/model1_EfficientNetB3_gen' + iteration)