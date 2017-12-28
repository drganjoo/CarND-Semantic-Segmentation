# Semantic Segmentation

### Introduction

The goal of the project is to identify parts of road in given images. Semantic Segmentation is to be carried out using Fully Convolutional Network (FCN).

#### Model Architecture

Pretrained VGG16 that already has 1x1 convolution layers has been used. As first step, the last layer is converted from 1x1x4096 to 1x1x2 (as there are 2 classes, road and not-road). The last layer is then upscaled by 2.

Pool4 layer from VGG16 is then converted to 1x1x2, added with last layer and then upscaled by 2.

Pool3 layer from VGG16 is then converted to 1x1x2, added with the Pool4+LastLayer, then upscaled by 8

#### Training Data

[Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) has been downloaded from [here](http://www.cvlibs.net/download.php?file=data_road.zip) and used for training. VGG16 pretrained model is download from [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)

#### Model Parameters

|Parameter Name|Value|
|--|--|
|Batch Size|10|
|Epochs|30|
|Learning Rate|0.0008|
|Optimizer|AdamOptimizer|
|Keep Probability| 0.5|

#### Loss Function

As suggested in the [Forum Post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100), I have included L2-Regularizer in the computation of loss function as well. The overall loss function:

```
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_constant = 1
    loss = cross_entropy + reg_constant * sum(reg_losses)

    cross_entropy_loss = tf.reduce_mean(loss)
```

Also, training has been limited to FCN layers only by passing the tensors to the AdamOptimizer as var_list

```
    collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fcn')
    training_operation = optimizer.minimize(cross_entropy_loss, var_list=collection)
```

#### Result

|Epoch|Loss|
|--|---|
|0 |12.8303590662|
|1 |5.7670175625|
|2| 3.23522536235|
|3| 1.96706364724|
|4| 1.34534895296|
|5| 1.02739088634|
|6| 0.842822642681|
|..|..|
|18|0.598858683373|
|19|0.596468009338|

#### Functions Used in Implementation

```load_vgg``` Loads the VGG16 pre trained network and returns the following tensors:

a) Input Image tensor   
b) Keep Probability tensor from the dropout layer   
c) Pool3    
d) Pool4   
e) Layer 7 output   

```layers``` adds the FCN layers on top of the VGG16 network. All nodes are scoped under ```fcn``` so that it is easier to pass them to the optimizer

```optimize``` defines the loss function using cross entropy and uses AdamOptimizer as the minimizer

```train_nn``` uses get_batches_fn, to train batches and then computes the loss for the Epoch. If it is the minimum loss so far, it is saved to './best.ckpt'

#### Run

Run the following command to run the project:
```
python main.py
```
#### Shortcomings

1) **Shortage of data**. There are only 289 sample images using, which the model has been trained. Ideally would have liked to have more or I wish I had the time to augment the images.

2) For each Epoch, training loss is printed but **validation loss has not been computed**. This is a big shortcoming, but since there were only 289 images, did not want to lower the training set by using only 20%

3) **Intermediate layer outputs have not been visualized** to get a better understanding of whats happening

4) **Training loss / validation loss has not been plotted**. This would have given a better understanding of how the model is progressing.

5) **Normalization was not gauranteed.** Ideally, the input data should be normalized before training on it
