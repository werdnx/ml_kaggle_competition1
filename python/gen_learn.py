from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from efficientnet.keras import EfficientNetB3
from keras import callbacks
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import cv2
import pandas as pd


def run():
    train_dir = '/input/jpeg/train512_nohair'
    test_dir = '../input/jpeg/test512_nohair'
    train_df = pd.read_csv('/input/train.csv')
    train_df['target'] = train_df['target'].astype('str')
    train_df['image_name'] = train_df['image_name'].astype(str) + '.jpg'
    train_df.head()

    train_datagen = ImageDataGenerator(
        rescale=1 / 255,
        validation_split=0.10,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=train_dir,
        x_col="image_name",
        y_col="target",
        target_size=(512, 512),
        subset="training",
        batch_size=2,
        shuffle=True,
        class_mode="binary"
    )
    val_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=train_dir,
        x_col="image_name",
        y_col="target",
        target_size=(512, 512),
        subset="validation",
        batch_size=2,
        shuffle=True,
        class_mode="binary"
    )
    efficient_net = EfficientNetB3(
        weights='imagenet',
        input_shape=(512, 512, 3),
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
    history = model.fit(
        train_generator,
        epochs=50,
        steps_per_epoch=15,
        validation_data=val_generator,
        validation_steps=7,
        verbose=1,
    )
    model.save('/output/model1_EfficientNetB3_gen')