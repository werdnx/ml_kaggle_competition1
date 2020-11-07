TRAIN_PATH = '/wdata/train'
PREPROCESS_PATH = '/wdata/preprocessed_audio/'
PREPROCESS_PATH_TEST = '/wdata/preprocessed_audio_test/'
MODEL_PATH = '/wdata/model/trained_model'
TEST_PATH = '/wdata/test'
SAMPLE_RATE = 16000
AUGMENT = False
FOLDS = 3
EPOCHS = 50
BATCH_SIZE_TRAIN = 128
BATCH_SIZE_TEST = 128
WINDOW = 180000
# SECONDS = 1

MODEL_PARAMS = [{'NAME': '10_sec', 'SECONDS': 10},
                {'NAME': '7_sec', 'SECONDS': 7},
                {'NAME': '5_sec', 'SECONDS': 5},
                {'NAME': '3_sec', 'SECONDS': 3},
                {'NAME': '1_sec', 'SECONDS': 1}]
