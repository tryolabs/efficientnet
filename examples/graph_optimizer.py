import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.platform import gfile

FROZEN_GRAPH_FILENAME = "../models/B0.pb"
TF_GPU_FRACTION = 0.67
TRT_GPU_BYTES = 3 << 29  # <= 33% of 4GB in Bytes

with tf.gfile.GFile(FROZEN_GRAPH_FILENAME, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

trt_graph = trt.create_inference_graph(
    input_graph_def=graph_def,  # frozen model
    outputs=["activation_27/Softmax"],
    max_batch_size=1,  # specify your max batch size
    max_workspace_size_bytes=TRT_GPU_BYTES,  # specify the max workspace
    precision_mode="FP16")  # precision, can be "FP32" (32 floating point precision) or "FP16"

# write the TensorRT model to be used later for inference
with gfile.FastGFile("../models/trt_B0.pb", 'wb') as f:
    f.write(trt_graph.SerializeToString())
print("TensorRT model is successfully stored!")

# check how many ops of the original frozen model
all_nodes = len([1 for n in graph_def.node])
print("numb. of all_nodes in frozen graph:", all_nodes)

# check how many ops that is converted to TensorRT engine
trt_engine_nodes = len(
    [1 for n in trt_graph.node if str(n.op) == 'TRTEngineOp'])
print("numb. of trt_engine_nodes in TensorRT graph:", trt_engine_nodes)
all_nodes = len([1 for n in trt_graph.node])
print("numb. of all_nodes in TensorRT graph:", all_nodes)
