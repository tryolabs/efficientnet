import os
import sys
import psutil as ps
from keras.applications.imagenet_utils import decode_predictions
from timeit import default_timer as timer

from efficientnet import EfficientNetB0
model_name = "B0_keras"
model = EfficientNetB0(weights="imagenet")

model.save(model_name + ".h5")
