import os
import sys
import timeit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from keras.applications.imagenet_utils import decode_predictions

from efficientnet import EfficientNetB3
from efficientnet import center_crop_and_resize, preprocess_input



# image = imread("../misc/panda.jpg")

# plt.figure(figsize=(10, 10))
# plt.imshow(image)
# plt.show()

model = EfficientNetB3(weights="imagenet")


class Params():
    image_size = model.input_shape[1]
    NUM_CLASSES = 1000  # classes
    NUM_IMAGES = 4  # images per folder
    cls = 0  # starting imagenet class
    im = 0  # pointer to current image of class i
    top_1_correct = 0
    total_imgs = 0


# preprocess image
def predict(p):
    img_file = "../../imagenetv2-matched-frequency/{}/{}.jpeg".format(p.cls, p.im)
    # print("Predict {}".format(img_file))
    image = imread(img_file)
    x = center_crop_and_resize(image, image_size=p.image_size)
    x = preprocess_input(x)
    x = np.expand_dims(x, 0)

    # make prediction and decode
    y = model.predict(x)
    pred_class = np.argmax(y)
    # print("Predicted class: {}".format(pred_class))
    # v = decode_predictions(y)

    # Evaluate accuracy
    p.total_imgs += 1.0
    p.top_1_correct += (pred_class == p.cls)

    # Loop through folders and images
    p.im += 1
    if p.im == p.NUM_IMAGES:
        p.cls = p.cls + 1 if p.cls < p.NUM_CLASSES - 1 else 0
        p.im = 0
        print("Entering class: {}".format(p.cls))


params = Params()

print(timeit.repeat('predict(params)', number=1, repeat=500, globals=globals()))
print("Accuracy: {:.2f}%".format(100 * params.top_1_correct/params.total_imgs))
