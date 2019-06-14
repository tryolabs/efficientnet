import os
import sys
import gc
import numpy as np
import matplotlib.pyplot as plt
import psutil as ps
from matplotlib.pyplot import imread
from keras.applications.imagenet_utils import decode_predictions
from timeit import default_timer as timer
from tqdm import tqdm

from efficientnet import EfficientNetB3
from efficientnet import center_crop_and_resize, preprocess_input


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# image = imread("../misc/panda.jpg")

# plt.figure(figsize=(10, 10))
# plt.imshow(image)
# plt.show()

NUM_CLASSES = 1000  # classes
NUM_IMAGES = 4  # images per folder
ITERATIONS = 103

print(f"loading model, memory info:{ps.virtual_memory()}")
model = EfficientNetB3(weights="imagenet")
print(f"model loaded, memory info:{ps.virtual_memory()}")


def predict():
    image_size = model.input_shape[1]
    top_1_correct = 0
    total_imgs = 0

    img_class = 0  # starting imagenet class
    img_num_in_class = 0  # pointer to current image of class i
    predictions_info = []

    total_img_classes = ITERATIONS // NUM_IMAGES
    total_img_classes += 1 if ITERATIONS % NUM_IMAGES != 0 else 0

    if total_img_classes > NUM_CLASSES:
        print("Some images will be predicted more than once")
        print("because ITERATIONS / NUM_IMAGES > NUM_CLASSES")

    for img_class in tqdm(range(total_img_classes)):
        gc.collect()
        # to avoid class overflow
        actual_class = img_class % NUM_CLASSES

        # we only want to process up until ITERATIONS times
        # if we are on the last class we don't process NUM_IMAGES images, only what's left.
        total_img_num_in_class = NUM_IMAGES if img_class < (
            ITERATIONS // NUM_IMAGES) else ITERATIONS % NUM_IMAGES

        for img_num_in_class in tqdm(range(total_img_num_in_class)):
            img_file = f"../../imagenetv2-matched-frequency/{actual_class}/{img_num_in_class}.jpeg"
            # print("Predict {}".format(img_file))
            image = imread(img_file)
            # preprocess image
            x = center_crop_and_resize(image, image_size=image_size)
            x = preprocess_input(x)
            x = np.expand_dims(x, 0)

            # make prediction and decode
            gc.disable()
            mem = ps.virtual_memory()
            start = timer()
            y = model.predict(x)
            end = timer()
            gc.enable()

            predictions_info.append(
                {"memory_info": mem, "prediction_time": round(end - start, 4)})

            pred_class = np.argmax(y)
            # print(f"Predicted class: {pred_class}")
            # v = decode_predictions(y)

            # Evaluate accuracy
            total_imgs += 1.0
            top_1_correct += (pred_class == actual_class)

    accuracy = (100 * top_1_correct/total_imgs)
    return predictions_info, accuracy


pred_info, accuracy = predict()
for info in pred_info:
    print(
        f"Memory in use: {info['memory_info'].percent}%, Time: {info['prediction_time']}")
print(f"Accuracy: {accuracy:.2f}%")
