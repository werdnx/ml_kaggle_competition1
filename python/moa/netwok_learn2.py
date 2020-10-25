import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import tensorflow_addons as tfa
from sklearn.model_selection import KFold

print('Tensorflow version:', tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 128
EPOCHS = 100
NFOLD = 5

#adam04
# {'dropout_rate': 0.25407056565585273, 'hidden_unit_1': 2, 'hidden_unit_2': 1, 'hidden_unit_3': 4}
# adam05
# {'dropout_rate': 0.41899471284989165, 'hidden_unit_1': 3, 'hidden_unit_2': 0, 'hidden_unit_3': 4}

top_feats = [0, 1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 14, 15,
             16, 18, 19, 20, 21, 23, 24, 25, 27, 28, 29, 30, 31,
             32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 44, 45, 46,
             48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
             63, 64, 65, 66, 68, 69, 70, 71, 72, 73, 74, 75, 76,
             78, 79, 80, 81, 82, 83, 84, 86, 87, 88, 89, 90, 92,
             93, 94, 95, 96, 97, 99, 100, 101, 103, 104, 105, 106, 107,
             108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
             121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 132, 133, 134,
             135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147,
             149, 150, 151, 152, 153, 154, 155, 157, 159, 160, 161, 163, 164,
             165, 166, 167, 168, 169, 170, 172, 173, 175, 176, 177, 178, 180,
             181, 182, 183, 184, 186, 187, 188, 189, 190, 191, 192, 193, 195,
             197, 198, 199, 202, 203, 205, 206, 208, 209, 210, 211, 212, 213,
             214, 215, 218, 219, 220, 221, 222, 224, 225, 227, 228, 229, 230,
             231, 232, 233, 234, 236, 238, 239, 240, 241, 242, 243, 244, 245,
             246, 248, 249, 250, 251, 253, 254, 255, 256, 257, 258, 259, 260,
             261, 263, 265, 266, 268, 270, 271, 272, 273, 275, 276, 277, 279,
             282, 283, 286, 287, 288, 289, 290, 294, 295, 296, 297, 299, 300,
             301, 302, 303, 304, 305, 306, 308, 309, 310, 311, 312, 313, 315,
             316, 317, 320, 321, 322, 324, 325, 326, 327, 328, 329, 330, 331,
             332, 333, 334, 335, 338, 339, 340, 341, 343, 344, 345, 346, 347,
             349, 350, 351, 352, 353, 355, 356, 357, 358, 359, 360, 361, 362,
             363, 364, 365, 366, 368, 369, 370, 371, 372, 374, 375, 376, 377,
             378, 379, 380, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391,
             392, 393, 394, 395, 397, 398, 399, 400, 401, 403, 405, 406, 407,
             408, 410, 411, 412, 413, 414, 415, 417, 418, 419, 420, 421, 422,
             423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435,
             436, 437, 438, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450,
             452, 453, 454, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465,
             466, 468, 469, 471, 472, 473, 474, 475, 476, 477, 478, 479, 482,
             483, 485, 486, 487, 488, 489, 491, 492, 494, 495, 496, 500, 501,
             502, 503, 505, 506, 507, 509, 510, 511, 512, 513, 514, 516, 517,
             518, 519, 521, 523, 525, 526, 527, 528, 529, 530, 531, 532, 533,
             534, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547,
             549, 550, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563,
             564, 565, 566, 567, 569, 570, 571, 572, 573, 574, 575, 577, 580,
             581, 582, 583, 586, 587, 590, 591, 592, 593, 595, 596, 597, 598,
             599, 600, 601, 602, 603, 605, 607, 608, 609, 611, 612, 613, 614,
             615, 616, 617, 619, 622, 623, 625, 627, 630, 631, 632, 633, 634,
             635, 637, 638, 639, 642, 643, 644, 645, 646, 647, 649, 650, 651,
             652, 654, 655, 658, 659, 660, 661, 662, 663, 664, 666, 667, 668,
             669, 670, 672, 674, 675, 676, 677, 678, 680, 681, 682, 684, 685,
             686, 687, 688, 689, 691, 692, 694, 695, 696, 697, 699, 700, 701,
             702, 703, 704, 705, 707, 708, 709, 711, 712, 713, 714, 715, 716,
             717, 723, 725, 727, 728, 729, 730, 731, 732, 734, 736, 737, 738,
             739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751,
             752, 753, 754, 755, 756, 758, 759, 760, 761, 762, 763, 764, 765,
             766, 767, 769, 770, 771, 772, 774, 775, 780, 781, 782, 783, 784,
             785, 787, 788, 790, 793, 795, 797, 799, 800, 801, 805, 808, 809,
             811, 812, 813, 816, 819, 820, 821, 822, 823, 825, 826, 827, 829,
             831, 832, 833, 834, 835, 837, 838, 839, 840, 841, 842, 844, 845,
             846, 847, 848, 850, 851, 852, 854, 855, 856, 858, 860, 861, 862,
             864, 867, 868, 870, 871, 873, 874]

print(len(top_feats))


def preprocess(df):
    df.loc[:, 'cp_type'] = df.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})
    df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})
    del df['sig_id']
    return df


def create_model(num_columns, hidden_units, dropout_rate):
    inp = tf.keras.layers.Input(shape=(num_columns,))
    x = tf.keras.layers.BatchNormalization()(inp)

    for units in hidden_units:
        x = tfa.layers.WeightNormalization(tf.keras.layers.Dense(units, activation='relu'))(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        x = tf.keras.layers.BatchNormalization()(x)

    out = tfa.layers.WeightNormalization(tf.keras.layers.Dense(206, activation='sigmoid'))(x)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=tfa.optimizers.Lookahead(tf.optimizers.Adam(1e-4)), loss='binary_crossentropy')
    return model


def evaluate(params, train_targets, train, test):
    hu = [params['hidden_unit_1'], params['hidden_unit_2']]

    if params['hidden_unit_3'] != 0:
        hu.append(params['hidden_unit_3'])

    p = {'hidden_units': hu,
         'dropout_rate': params['dropout_rate'],
         }

    res_nn = train_targets.copy()
    res_nn.loc[:, train_targets.columns] = 0
    pe = np.zeros((test.shape[0], 206))
    for n, (tr, te) in enumerate(KFold(n_splits=NFOLD,
                                       random_state=0,
                                       shuffle=True).split(train_targets)):
        #         print(f'Fold {n}:')

        x_tr, x_val = train.values[tr][:, top_feats], train.values[te][:, top_feats]
        y_tr, y_val = train_targets.astype(float).values[tr], train_targets.astype(float).values[te]

        model = create_model(len(top_feats), p['hidden_units'], p['dropout_rate'])

        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3,
                                verbose=0, epsilon=1e-4, mode='min')

        ckp = ModelCheckpoint('model_' + str(n) + '.hdf5', monitor='val_loss', verbose=0,
                              save_best_only=True, save_weights_only=True, mode='min')

        es = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, mode='min',
                           baseline=None, restore_best_weights=True, verbose=0)

        model.fit(x_tr, y_tr, validation_data=(x_val, y_val),
                  epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[rlr, ckp, es], verbose=1)
        pe += model.predict(test.values[:][:, top_feats], batch_size=BATCH_SIZE, verbose=1) / NFOLD

    return pe


def main():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
        print('Running on TPU ', tpu.master())
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        strategy = tf.distribute.get_strategy()  # default distribution strategy in Tensorflow. Works on CPU and single GPU.

    print("REPLICAS: ", strategy.num_replicas_in_sync)
    MIXED_PRECISION = False
    XLA_ACCELERATE = True

    if MIXED_PRECISION:
        from tensorflow.keras.mixed_precision import experimental as mixed_precision
        if tpu:
            policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
        else:
            policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
        print('Mixed precision enabled')

    if XLA_ACCELERATE:
        tf.config.optimizer.set_jit(True)
        print('Accelerated Linear Algebra enabled')

    train_features = pd.read_csv('../input/train_features.csv')
    train_targets = pd.read_csv('../input/train_targets_scored.csv')
    test_features = pd.read_csv('../input/test_features.csv')

    ss = pd.read_csv('../input/sample_submission.csv')
    train = preprocess(train_features)
    test = preprocess(test_features)

    del train_targets['sig_id']
    train.head()
    # params = {'dropout_rate': 0.4206650042019096, 'hidden_unit_1': 1024, 'hidden_unit_2': 6144, 'hidden_unit_3': 0}
    # params = {'dropout_rate': 0.25407056565585273, 'hidden_unit_1': 2048, 'hidden_unit_2': 4096, 'hidden_unit_3': 0}
    params = {'dropout_rate': 0.41899471284989165, 'hidden_unit_1': 1024, 'hidden_unit_2': 6144, 'hidden_unit_3': 0}
    result = evaluate(params, train_targets, train, test)
    columns = pd.read_csv('../input/train_targets_scored.csv')
    del columns['sig_id']
    sub = pd.DataFrame(data=result, columns=columns.columns)
    sample = pd.read_csv('../input/sample_submission.csv')
    sub.insert(0, column='sig_id', value=sample['sig_id'])
    sub.to_csv('/output/submission_net2_adam05_v5.csv', index=False)


# Add scaler ?


if __name__ == "__main__":
    main()
