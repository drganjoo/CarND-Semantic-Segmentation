import tensorflow as tf
import helper
import project_tests as tests
import os

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
    #graph = tf.get_default_graph()
    graph = sess.graph

    # test_writer = tf.summary.FileWriter('logs')
    # test_writer.add_graph(graph)

    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    print(layer7.get_shape())
    print(layer4.get_shape())

    print(tf.trainable_variables())

    frozen_graph = tf.graph_util.convert_variables_to_constants(
        sess,  # The session is used to retrieve the weights
        graph.as_graph_def(),  # The graph_def is used to retrieve the nodes
        ['layer7_out']
    )

    s = tf.shape(layer7)
    print(sess.run(s, feed_dict={keep_prob : 0.5}))


    print('done')