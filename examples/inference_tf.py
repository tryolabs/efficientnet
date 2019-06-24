import os
import sys
import gc
import numpy as np
import psutil as ps
import tensorflow as tf
from matplotlib.pyplot import imread

from timeit import default_timer as timer
from tqdm import tqdm

from efficientnet import EfficientNetB3
from efficientnet import center_crop_and_resize, preprocess_input


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
GPU_FRACTION = 1

NUM_CLASSES = 1000  # classes
NUM_IMAGES = 4  # images per folder
ITERATIONS = 103

FROZEN_GRAPH_FILENAME = "../models/B0.pb"


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")

    return graph


def predict():
    # We use our "load_graph" function
    graph = load_graph(FROZEN_GRAPH_FILENAME)

    # We access the input and output nodes
    x = graph.get_tensor_by_name(graph.get_operations()[0].name + ":0")
    y = graph.get_tensor_by_name(graph.get_operations()[-1].name + ":0")

    # image_size = model.input_shape[1]
    image_size = x.get_shape().as_list()[1]
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

    # We launch a Session

    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=GPU_FRACTION)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(graph=graph, config=sess_config) as sess:

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
                x_in = center_crop_and_resize(image, image_size=image_size)
                x_in = preprocess_input(x_in)
                x_in = np.expand_dims(x_in, 0)

                # make prediction and decode
                gc.disable()
                mem = ps.virtual_memory()
                start = timer()
                y_out = sess.run(y, feed_dict={
                    x: x_in
                })
                end = timer()
                gc.enable()

                predictions_info.append(
                    {"memory_info": mem, "prediction_time": round(end - start, 4)})

                pred_class = np.argmax(y_out)
                # print(f"Predicted class: {pred_class}")

                # Evaluate accuracy
                total_imgs += 1.0
                top_1_correct += (pred_class == actual_class)

    accuracy = (100 * top_1_correct/total_imgs)
    return predictions_info, accuracy


pred_info, accuracy = predict()
# get model file name and substract extention
model_name = "".join(FROZEN_GRAPH_FILENAME.split("/")[-1].split(".")[:-1])
with open(f"last_results_{model_name}.txt", "w+") as f:
    for info in pred_info[2:]:
        f.write(str(info['prediction_time']) + "\n")
        print(
            f"Memory in use: {info['memory_info'].percent}%, Time: {info['prediction_time']}")
    accuracy_str = f"Accuracy: {accuracy:.2f}% \n"
    f.write(accuracy_str)
    print(accuracy_str)
