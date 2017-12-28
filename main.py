import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import numpy as np

save_best = False

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    graph = tf.get_default_graph()

    # for n in tf.get_default_graph().as_graph_def().node:
    #     print(n.name)

    # for op in graph.get_operations():
    #     print(op.name)

    node_names = ['image_input:0', 'keep_prob:0', 'layer3_out:0', 'layer4_out:0','layer7_out:0']
    nodes = [graph.get_tensor_by_name(name) for name in node_names]

    return nodes

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    with tf.variable_scope("fcn"):
        regularizer = tf.contrib.layers.l2_regularizer(1e-3)
        initializer = tf.random_normal_initializer(stddev=0.01)

        layer7_1x1 = tf.layers.conv2d(vgg_layer7_out,
                                      num_classes,
                                      1,
                                      padding = 'same',
                                      kernel_initializer = initializer,
                                      kernel_regularizer = regularizer,
                                      name = 'layer7_1x1')

        layer7_1x1_upscaled = tf.layers.conv2d_transpose(layer7_1x1,
                                                         num_classes,
                                                         4,
                                                         2,
                                                         padding = 'same',
                                                         kernel_initializer=initializer,
                                                         kernel_regularizer=regularizer,
                                                         name='layer7_1x1_upscaled')

        # skip connection (add pool4 with upscaled layer 7)

        pool4_scaled = tf.multiply(vgg_layer4_out,
                                   0.01,
                                   name='pool4_scaled')
        pool4_1x1 = tf.layers.conv2d(pool4_scaled,
                                     num_classes,
                                     1,
                                     padding = 'same',
                                     kernel_initializer=initializer,
                                     kernel_regularizer=regularizer,
                                     name='pool4_1x1')

        l7p4 = tf.add(layer7_1x1_upscaled,
                      pool4_1x1,
                      name='layer7_pool4_skip')

        # again upscale by 2

        l7p4_upscaled = tf.layers.conv2d_transpose(l7p4,
                                                   num_classes,
                                                   4,
                                                   2,
                                                   padding = 'same',
                                                   kernel_initializer=initializer,
                                                   kernel_regularizer=regularizer,
                                                   name='l7p4_upscaled')

        # skip connection, add pool3 with upscaled skipped l7+p4
        pool3_scaled = tf.multiply(vgg_layer3_out,
                                   0.0001,
                                   name='pool3_scaled')

        pool3_1x1 = tf.layers.conv2d(pool3_scaled,
                                     num_classes,
                                     1,
                                     padding='same',
                                     kernel_initializer=initializer,
                                     kernel_regularizer=regularizer,
                                     name='pool3_1x1')

        l7p4p3 = tf.add(l7p4_upscaled,
                        pool3_1x1,
                        name='layer7_pool4_pool3_skip')

        l7p4p3_upscaled = tf.layers.conv2d_transpose(l7p4p3,
                                                     num_classes,
                                                     16,
                                                     8,
                                                     padding = 'same',
                                                     kernel_initializer=initializer,
                                                     kernel_regularizer=regularizer,
                                                     name='l7p4p3_upscaled')

    return l7p4p3_upscaled

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    # https://stackoverflow.com/questions/46615623/do-we-need-to-add-the-regularization-loss-into-the-total-loss-in-tensorflow-mode

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_constant = 1
    loss = cross_entropy + reg_constant * sum(reg_losses)
    # loss = cross_entropy

    cross_entropy_loss = tf.reduce_mean(loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # only allow the new layers to be trained and nothing in VGG16 to be trainable
    collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fcn')

    if len(collection) > 0:
        training_operation = optimizer.minimize(cross_entropy_loss, var_list=collection)
    else:
        # for test cases 'fcn' scoped layers are not defined
        training_operation = optimizer.minimize(cross_entropy_loss)

    return logits, training_operation, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):

    global save_best
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    if save_best:
        saver = tf.train.Saver()
        best_filename = './best.ckpt'
        lowest_score = 9999

    test_writer = tf.summary.FileWriter('logs')
    test_writer.add_graph(sess.graph)

    print('Training begins...')
    print('|Epoch|Loss|')
    print('|--|--|')

    for epoch in range(epochs):
        batch_no = 1
        for batch_images, batch_labels in get_batches_fn(batch_size):
            # print('Batch #: {} Batch Size: {} Label Size: {}'.format(batch_no, batch_images.shape, batch_labels.shape))

            sess.run(train_op, feed_dict={input_image : batch_images,
                                          correct_label: batch_labels,
                                          keep_prob: 0.5,
                                          learning_rate: 0.0008})

            batch_no += 1

        # compute loss
        n_data = 0
        loss = 0

        for batch_images, batch_labels in get_batches_fn(batch_size):
            batch_loss = sess.run(cross_entropy_loss, feed_dict={input_image : batch_images,
                                          correct_label: batch_labels,
                                          keep_prob: 1})
            n_batch = batch_images.shape[0]
            loss += (batch_loss * n_batch)
            n_data += n_batch
            # print(n_batch, n_data)

        loss /= n_data

        print('|{}|{:4f}|'.format(epoch, loss))

        if save_best and loss < lowest_score:
            # print('Found lowest loss: ', loss, 'saving!!!!')
            saver.save(sess, best_filename)

tests.test_train_nn(train_nn)


def print_trainable():
    print('~' * 100)
    print('Trainable variables:')
    print(tf.trainable_variables())
    print('~' * 100)


def run():
    global save_best

    epochs = 50
    batch_size = 8
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    # print('-' * 100)
    # print('Running training session')
    # print('-' * 100)

    save_best = True
    tf.reset_default_graph()

    with tf.Session() as sess:
        vgg_path = os.path.join(data_dir, 'vgg')

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        input_image, keep_prob, layer3_out, layer4_out, layer7_out  = load_vgg(sess, vgg_path)
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)

        correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes], name = 'correct_label')
        learning_rate = tf.placeholder(tf.float32, shape=[], name = 'learning_rate')

        logits, training_operation, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes)

        # graph = tf.get_default_graph()
        # print(graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

        ## use tensorboard to save the graph and view it
        test_writer = tf.summary.FileWriter('logs')
        test_writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer())

        train_nn(sess, epochs, batch_size, get_batches_fn, training_operation,
                 cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
