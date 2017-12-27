import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

save_best = False

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))



def load_graph_nodes(graph, node_names):
    nodes = [graph.get_tensor_by_name(name) for name in node_names]
    return nodes
    #
    # w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    # keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    # layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    # layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    # layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    #
    # return w1, keep_prob, layer3, layer4, layer7, graph

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights

    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    graph = tf.get_default_graph()
    nodes = ['image_input:0', 'keep_prob:0', 'layer3_out:0', 'layer4_out:0','layer7_out:0']
    return load_graph_nodes(graph, nodes)

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
    regularizer = tf.contrib.layers.l2_regularizer(1e-3)

    #conv_1_1 = vgg_layer7_out
    conv_one_by_one = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding = 'same',
                                 kernel_regularizer=regularizer)
    conv_1x1_upscaled = tf.layers.conv2d_transpose(conv_one_by_one, num_classes, 4, 2, padding = 'same',
                                kernel_regularizer=regularizer)


    pool4_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding = 'same',
                                 kernel_regularizer=regularizer)

    # print(vgg_layer7_out.get_shape())
    # print(conv_1x1.get_shape())
    # print(conv_1x1_upscaled.get_shape())
    # print(pool4_1x1.get_shape())

    # tf.Print(pool4_1x1, [tf.shape(conv_1x1_upscaled), tf.shape(pool4_1x1)])

    c1x1_p4 = tf.add(conv_1x1_upscaled , pool4_1x1)
    c1x1_p4_upscaled = tf.layers.conv2d_transpose(c1x1_p4, num_classes, 4, 2, padding = 'same',
                                kernel_regularizer=regularizer)

    pool3_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',
                             kernel_regularizer=regularizer)
    c1x1_p4_p3 = tf.add(c1x1_p4_upscaled, pool3_1x1)
    output = tf.layers.conv2d_transpose(c1x1_p4_p3, num_classes, 16, 8, padding = 'same',
                                kernel_regularizer=regularizer)

    #tf.Print(output, tf.shape(output))
    print('conv_1x1: ', conv_one_by_one)
    print('conv_1x1 Upscaled: ', conv_1x1_upscaled)
    print('pool4_1x1: ', pool4_1x1)
    print('c1x1_p4: ', c1x1_p4)
    print('c1x1_p4_upscaled: ', c1x1_p4_upscaled)
    print('pool3_1x1: ', pool3_1x1)
    print('c1x1_p4_p3: ', c1x1_p4_p3)
    print('output: ', output)
    return output

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

    # logits = nn_last_layer

    print('Logits before shape:', nn_last_layer.get_shape())
    print('Logits Shape: ', logits)
    print('Correct label', correct_label)
    print('learning_rate', learning_rate)

    #logits = nn_last_layer

    # tf.Print(tf.shape(logits))

    # https://stackoverflow.com/questions/46615623/do-we-need-to-add-the-regularization-loss-into-the-total-loss-in-tensorflow-mode


    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label)
    # reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # reg_constant = 1
    # loss = cross_entropy + reg_constant * sum(reg_losses)
    loss = cross_entropy

    cross_entropy_loss = tf.reduce_mean(loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
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

    # test_writer = tf.summary.FileWriter('logs')
    # test_writer.add_graph(sess.graph)

    for epoch in range(epochs):
        print('Epoch: ', epoch)

        for batch_images, batch_labels in get_batches_fn(batch_size):
            print('Batch: ', batch_images.shape)
            print('Label Size: ', batch_labels.shape)

            sess.run(train_op, feed_dict={input_image : batch_images,
                                          correct_label: batch_labels,
                                          keep_prob: 0.5,
                                          learning_rate: 0.0008})

        loss = sess.run(cross_entropy_loss)
        print('Loss:', loss)

        if save_best and loss < lowest_score:
            print('Found lowest loss: ', loss, 'saving!!!!')
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
    frozen_graph = data_dir + '/frozen.pb'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    print('-' * 100)
    print('Running training session')
    print('-' * 100)

    save_best = True
    tf.reset_default_graph()

    with tf.Session() as sess:
        vgg_path = os.path.join(data_dir, 'vgg')

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        input_image, keep_prob, layer3_out, layer4_out, layer7_out, graph = load_vgg(sess, vgg_path)

        print_trainable()
        print('Freezing graph')

        frozen_node_names = ['image_input', 'keep_prob', 'layer3_out', 'layer4_out', 'layer7_out']
        graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            graph.as_graph_def(),
            frozen_node_names
        )

        # with tf.gfile.GFile(frozen_graph, "wb") as f:
        #     f.write(graph_def.SerializeToString())

    tf.reset_default_graph()

    # with tf.gfile.GFile(frozen_graph, "rb") as f:
    #     graph_def = tf.GraphDef()
    #     graph_def.ParseFromString(f.read())

    with tf.Session() as sess:
        tf.import_graph_def(graph_def)

        graph = tf.get_default_graph()

        print('Using frozen graph')
        print_trainable()

        for n in graph_def.node:
            print(n.name)

        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_graph_nodes(graph, frozen_node_names)
        layer_output = layers(layer3_out, layer4_out, layer7_out, num_classes)

        # TODO: Train NN using the train_nn function
        correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes], name = 'correct_label')
        learning_rate = tf.placeholder(tf.float32, shape=[], name = 'learning_rate')

        logits, training_operation, cross_entropy_loss = optimize(layer_output, correct_label, learning_rate, num_classes)

        # graph = tf.get_default_graph()
        # print(graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

        ## use tensorboard to save the graph and view it
        # test_writer = tf.summary.FileWriter('logs')
        # test_writer.add_graph(sess.graph)

        sess.run(tf.global_variables_initializer())


        train_nn(sess, epochs, batch_size, get_batches_fn, training_operation,
                 cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
