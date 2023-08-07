import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config
import utils
from pathlib import Path

show_image = False

# load dataset
train_set = pd.read_csv(config.PATH_TRAIN)
test_set = pd.read_csv(config.PATH_TEST)

# shuffle dataset
train_set = train_set.sample(frac=1, random_state=config.RANDOM_STATE)

# show image
fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(10, 5))
if show_image:
    for i, row in enumerate(train_set.iloc[:10].iterrows()):
        label = row[1]["label"]
        label_str = utils.label_dict.get(label)
        features = row[1][1:]

        img = np.array(features).reshape((28, 28))
        axs.flat[i].imshow(img, cmap="gray")
        axs.flat[i].set_title(label_str)

    plt.show()

# rescale image to 0-1
train_set = np.divide(train_set, 255)
test_set = np.divide(test_set, 255)

# save dataset as parquet
if not Path(config.PATH_TRAIN_PROCESSED).exists():
    train_set.to_parquet(config.PATH_TRAIN_PROCESSED)
    test_set.to_parquet(config.PATH_TEST_PROCESSED)
    print('file saved')
else:
    print('file exists')