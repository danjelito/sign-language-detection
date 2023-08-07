from pathlib import Path

RANDOM_STATE= 8

PATH_TRAIN = Path.cwd() / 'input/sign_mnist_train.csv'
PATH_TEST = Path.cwd() / 'input/sign_mnist_test.csv'

PATH_TRAIN_PROCESSED = Path.cwd() / 'input/sign_mnist_train.parquet'
PATH_TEST_PROCESSED = Path.cwd() / 'input/sign_mnist_test.parquet'

PATH_HAND_LANDMARKER = Path.cwd() / 'input/hand_landmarker.task'