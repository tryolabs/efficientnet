import tensorflow as tf
import sys

from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat

GRAPH_PB_PATH = 'jetson_B3_model.pb'

with tf.Session() as persisted_sess:
    print("load graph")
    with tf.gfile.GFile(GRAPH_PB_PATH, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        persisted_sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        writer = tf.summary.FileWriter(
            "./tf_summary", graph=persisted_sess.graph)
        # Print all operation names
        for op in persisted_sess.graph.get_operations():
            print(op.name)
        # next: do the following in bash:
        # tensorboard --logdir ./tf_summary/
