from keras.preprocessing.image import ImageDataGenerator
from efficientnet.keras import EfficientNetB3
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, GlobalMaxPooling2D, BatchNormalization
from keras.models import Sequential
import pandas as pd
import time

from python.old.ensemble.model_params import batch_size, target_size_, batch_size_, epochs_, model_name

CLASSES = 2


def run():
    train_dir = '/input/jpeg/train512_nohair_croped'
    train_df = pd.read_csv('/input/train.csv')
    steps_per_epoch_ = len(train_df) / batch_size
    validation_steps_ = steps_per_epoch_ / 5
    train_df['target'] = train_df['target'].astype('str')
    train_df['image_name'] = train_df['image_name'].astype(str)

    add_df = train_df.copy()
    add_df = add_df.iloc[0:0]
    for i in range(0, 60):
        pos_df_new = train_df[train_df['target'] == '1'].copy()
        pos_df_new['image_name'] = pos_df_new['image_name'] + '_' + str(i)
        frames = [add_df, pos_df_new]
        add_df = pd.concat(frames)

    frames = [train_df, add_df]
    train_df = pd.concat(frames)
    train_df['image_name'] = train_df['image_name'] + '.jpg'
    train_df = train_df.sample(frac=1)
    print (train_df.head())
    print("len of target = 1: " + str(len(train_df[train_df["target"] == '1'])))
    print("len of target = 0: " + str(len(train_df[train_df["target"] == '0'])))
    time.sleep(10)

    train_datagen = ImageDataGenerator(
        rescale=1 / 255,
        validation_split=0.10,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.2, 1.0],
        # zoom_range=[0.5, 1.0],
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
        batch_size=batch_size_,
        shuffle=True,
        class_mode="categorical"
    )
    val_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=train_dir,
        x_col="image_name",
        y_col="target",
        target_size=(target_size_, target_size_),
        subset="validation",
        batch_size=batch_size_,
        shuffle=True,
        class_mode="categorical"
    )
    efficient_net = EfficientNetB3(
        weights='imagenet',
        #weights=None,
        input_shape=(target_size_, target_size_, 3),
        include_top=False
        # pooling='max'
    )
    model = Sequential()
    model.add(efficient_net)
    model.add(GlobalMaxPooling2D())
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(CLASSES, activation='softmax'))
    # model = load_model('/output/model1_EfficientNetB3_gen' + '7')
    model.summary()
    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    # steps_per_epoch = train_size/ batches
    history = model.fit(
        train_generator,
        epochs=epochs_,
        steps_per_epoch=steps_per_epoch_,
        validation_data=val_generator,
        validation_steps=validation_steps_,
        verbose=1,
    )
    model.save('/output/' + model_name)
