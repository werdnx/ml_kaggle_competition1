TRAIN_PATH = '/wdata/train'
PREPROCESS_PATH = '/wdata/preprocessed_audio/'
PREPROCESS_PATH_TEST = '/wdata/preprocessed_audio_test/'
MODEL_PATH = '/wdata/model/trained_model'
TEST_PATH = '/wdata/test'
SAMPLE_RATE = 16000
AUGMENT = False
FOLDS = 3
EPOCHS = 50
# BATCH_SIZE_TRAIN = 64
# BATCH_SIZE_TEST = 32
WINDOW = 180000
# SECONDS = 1
DROPOUT = 0.1

MODEL_PARAMS = [
    {'NAME': '10_sec_b6', 'SECONDS': 10, 'TYPE': 'efficientnet-b6', 'TRAIN_BATCH': 16, 'VALID_BATCH': 8, 'EPOCHS': 40},
    {'NAME': '7_sec_b6', 'SECONDS': 7, 'TYPE': 'efficientnet-b6', 'TRAIN_BATCH': 16, 'VALID_BATCH': 8, 'EPOCHS': 40},
    {'NAME': '5_sec_b6', 'SECONDS': 5, 'TYPE': 'efficientnet-b6', 'TRAIN_BATCH': 16, 'VALID_BATCH': 8, 'EPOCHS': 40},
    {'NAME': '3_sec_b6', 'SECONDS': 3, 'TYPE': 'efficientnet-b6', 'TRAIN_BATCH': 16, 'VALID_BATCH': 8, 'EPOCHS': 40},
    {'NAME': '1_sec_b6', 'SECONDS': 1, 'TYPE': 'efficientnet-b6', 'TRAIN_BATCH': 16, 'VALID_BATCH': 8, 'EPOCHS': 40},
    {'NAME': '10_sec_b5', 'SECONDS': 10, 'TYPE': 'efficientnet-b5', 'TRAIN_BATCH': 16, 'VALID_BATCH': 8, 'EPOCHS': 40},
    {'NAME': '7_sec_b5', 'SECONDS': 7, 'TYPE': 'efficientnet-b5', 'TRAIN_BATCH': 16, 'VALID_BATCH': 8, 'EPOCHS': 40},
    {'NAME': '5_sec_b5', 'SECONDS': 5, 'TYPE': 'efficientnet-b5', 'TRAIN_BATCH': 16, 'VALID_BATCH': 8, 'EPOCHS': 40},
    {'NAME': '3_sec_b5', 'SECONDS': 3, 'TYPE': 'efficientnet-b5', 'TRAIN_BATCH': 16, 'VALID_BATCH': 8, 'EPOCHS': 40},
    {'NAME': '1_sec_b5', 'SECONDS': 1, 'TYPE': 'efficientnet-b5', 'TRAIN_BATCH': 16, 'VALID_BATCH': 8, 'EPOCHS': 40},
    {'NAME': '10_sec_b4', 'SECONDS': 10, 'TYPE': 'efficientnet-b4', 'TRAIN_BATCH': 32, 'VALID_BATCH': 16, 'EPOCHS': 50},
    {'NAME': '7_sec_b4', 'SECONDS': 7, 'TYPE': 'efficientnet-b4', 'TRAIN_BATCH': 32, 'VALID_BATCH': 16, 'EPOCHS': 50},
    {'NAME': '5_sec_b4', 'SECONDS': 5, 'TYPE': 'efficientnet-b4', 'TRAIN_BATCH': 32, 'VALID_BATCH': 16, 'EPOCHS': 50},
    {'NAME': '3_sec_b4', 'SECONDS': 3, 'TYPE': 'efficientnet-b4', 'TRAIN_BATCH': 32, 'VALID_BATCH': 16, 'EPOCHS': 50},
    {'NAME': '1_sec_b4', 'SECONDS': 1, 'TYPE': 'efficientnet-b4', 'TRAIN_BATCH': 32, 'VALID_BATCH': 16, 'EPOCHS': 50},
    {'NAME': '10_sec_b3', 'SECONDS': 10, 'TYPE': 'efficientnet-b3', 'TRAIN_BATCH': 32, 'VALID_BATCH': 16, 'EPOCHS': 50},
    {'NAME': '7_sec_b3', 'SECONDS': 7, 'TYPE': 'efficientnet-b3', 'TRAIN_BATCH': 32, 'VALID_BATCH': 16, 'EPOCHS': 50},
    {'NAME': '5_sec_b3', 'SECONDS': 5, 'TYPE': 'efficientnet-b3', 'TRAIN_BATCH': 32, 'VALID_BATCH': 16, 'EPOCHS': 50},
    {'NAME': '3_sec_b3', 'SECONDS': 3, 'TYPE': 'efficientnet-b3', 'TRAIN_BATCH': 32, 'VALID_BATCH': 16, 'EPOCHS': 50},
    {'NAME': '1_sec_b3', 'SECONDS': 1, 'TYPE': 'efficientnet-b3', 'TRAIN_BATCH': 32, 'VALID_BATCH': 16, 'EPOCHS': 50},
    {'NAME': '10_sec_b2', 'SECONDS': 10, 'TYPE': 'efficientnet-b2', 'TRAIN_BATCH': 32, 'VALID_BATCH': 16, 'EPOCHS': 50},
    {'NAME': '7_sec_b2', 'SECONDS': 7, 'TYPE': 'efficientnet-b2', 'TRAIN_BATCH': 32, 'VALID_BATCH': 16, 'EPOCHS': 50},
    {'NAME': '5_sec_b2', 'SECONDS': 5, 'TYPE': 'efficientnet-b2', 'TRAIN_BATCH': 32, 'VALID_BATCH': 16, 'EPOCHS': 50},
    {'NAME': '3_sec_b2', 'SECONDS': 3, 'TYPE': 'efficientnet-b2', 'TRAIN_BATCH': 32, 'VALID_BATCH': 16, 'EPOCHS': 50},
    {'NAME': '1_sec_b2', 'SECONDS': 1, 'TYPE': 'efficientnet-b2', 'TRAIN_BATCH': 32, 'VALID_BATCH': 16, 'EPOCHS': 50},
    {'NAME': '10_sec_b1', 'SECONDS': 10, 'TYPE': 'efficientnet-b1', 'TRAIN_BATCH': 64, 'VALID_BATCH': 32, 'EPOCHS': 50},
    {'NAME': '7_sec_b1', 'SECONDS': 7, 'TYPE': 'efficientnet-b1', 'TRAIN_BATCH': 64, 'VALID_BATCH': 32, 'EPOCHS': 50},
    {'NAME': '5_sec_b1', 'SECONDS': 5, 'TYPE': 'efficientnet-b1', 'TRAIN_BATCH': 64, 'VALID_BATCH': 32, 'EPOCHS': 50},
    {'NAME': '3_sec_b1', 'SECONDS': 3, 'TYPE': 'efficientnet-b1', 'TRAIN_BATCH': 64, 'VALID_BATCH': 32, 'EPOCHS': 50},
    {'NAME': '1_sec_b1', 'SECONDS': 1, 'TYPE': 'efficientnet-b1', 'TRAIN_BATCH': 64, 'VALID_BATCH': 32, 'EPOCHS': 50},
    {'NAME': '10_sec_b0', 'SECONDS': 10, 'TYPE': 'efficientnet-b0', 'TRAIN_BATCH': 64, 'VALID_BATCH': 32,
     'EPOCHS': 100},
    {'NAME': '7_sec_b0', 'SECONDS': 7, 'TYPE': 'efficientnet-b0', 'TRAIN_BATCH': 64, 'VALID_BATCH': 32, 'EPOCHS': 100},
    {'NAME': '5_sec_b0', 'SECONDS': 5, 'TYPE': 'efficientnet-b0', 'TRAIN_BATCH': 64, 'VALID_BATCH': 32, 'EPOCHS': 100},
    {'NAME': '3_sec_b0', 'SECONDS': 3, 'TYPE': 'efficientnet-b0', 'TRAIN_BATCH': 64, 'VALID_BATCH': 32, 'EPOCHS': 100},
    {'NAME': '1_sec_b0', 'SECONDS': 1, 'TYPE': 'efficientnet-b0', 'TRAIN_BATCH': 64, 'VALID_BATCH': 32, 'EPOCHS': 100}]
