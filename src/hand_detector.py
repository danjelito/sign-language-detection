import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import numpy as np
import cv2
import config
import utils
from PIL import Image

# create landmarker object
base_options = python.BaseOptions(model_asset_path=config.PATH_HAND_LANDMARKER)
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# load image
train_set = pd.read_csv(config.PATH_TRAIN)
image = train_set.iloc[0][1:]
image = image.values.reshape((28, 28)).astype(np.uint8)
image = mp.Image(data= image, image_format= mp.ImageFormat.GRAY8)

# detect landmark
detection_result = detector.detect(image)

# visualize
annotated_image = utils.draw_landmarks_on_image(image.numpy_view(), detection_result)
cv2.imshow('frame', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

cv2.waitKey(0)
cv2.destroyAllWindows()