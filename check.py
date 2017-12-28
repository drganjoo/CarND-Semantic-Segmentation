import tensorflow as tf
import helper
import project_tests as tests
import os
from tensorflow.core.protobuf import saved_model_pb2

num_classes = 2
image_shape = (160, 576)
data_dir = './data'
runs_dir = './runs'
tests.test_for_kitti_dataset(data_dir)

vgg_filename = 'vgg.zip'
vgg_path = os.path.join(data_dir, 'vgg')
vgg_files = [
    os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
    os.path.join(vgg_path, 'variables/variables.index'),
    os.path.join(vgg_path, 'saved_model.pb')]

missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
if missing_vgg_files:
    print('Download vgg first')
    exit(100)

helper.maybe_download_pretrained_vgg(data_dir)

vgg_tag = 'vgg16'

with tf.Session() as sess:
    vgg_path = os.path.join(data_dir, 'vgg')
    print(vgg_path)

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = sess.graph

    for op in graph.get_operations():
        print(op.name)

    test_writer = tf.summary.FileWriter('logs')
    test_writer.add_graph(sess.graph)