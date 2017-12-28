[//]: # (Image References)
[tensorboard_vgg16]: ./images/tensorboard-vgg16.png
[tensorboard_fcn]: ./images/tensorboard-fcn.png
[tensorboard_fcn_expanded]: ./images/tensorboard-fcn-expanded.png
[r1]: ./images/r1.png
[r2]: ./images/r2.png
[r3]: ./images/r3.png
[r4]: ./images/r4.png
[r5]: ./images/r5.png
[r6]: ./images/r6.png

# Semantic Segmentation

### Introduction

The goal of the project is to identify parts of road in given images. Semantic Segmentation is to be carried out using Fully Convolutional Network (FCN).

#### Model Architecture

Pretrained VGG16 that already has 1x1 convolution layers has been used. As first step, the last layer is converted from 1x1x4096 to 1x1x2 (as there are 2 classes, road and not-road). The last layer is then upscaled by 2.

Pool4 layer from VGG16 is then scaled by 0.01, converted to 1x1x2 and then added with last layer and then upscaled by 2.

Pool3 layer from VGG16 is then scaled by 0.0001, converted to 1x1x2 and then added with the Pool4+LastLayer. It is then upscaled by 8.

Pool3 and Pool4 have been scaled:
```
        pool4_scaled = tf.multiply(vgg_layer4_out,
                                   0.01,
                                   name='pool4_scaled')
        pool3_scaled = tf.multiply(vgg_layer3_out,
                                   0.0001,
                                   name='pool3_scaled')
```

Pre Trained VGG16:

![tensorboard_vgg16]

Layers added for FCN:

![tensorboard_fcn]

Expanded View of FCN:

![tensorboard_fcn_expanded]

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
|--|--|
|0|0.858214|
|1|0.727993|
|2|0.580164|
|3|0.561139|
|4|0.545859|
|5|0.532650|
|6|0.526948|
|7|0.519746|
|8|0.515383|
|9|0.512144|
|10|0.514359|
|11|0.504888|
|12|0.503120|
|13|0.506398|
|14|0.513169|
|15|0.497171|
|16|0.501403|
|17|0.495156|
|18|0.493737|
|19|0.498202|
|20|0.492219|
|21|0.493314|
|22|0.503448|
|23|0.489847|
|24|0.488712|
|25|0.489426|
|26|0.487190|
|27|0.493990|
|28|0.487113|
|29|0.486558|
|30|0.485502|
|31|0.487570|
|32|0.486182|
|33|0.484172|
|34|0.490326|
|35|0.484246|
|36|0.484060|
|37|0.483559|
|38|0.484606|
|39|0.483311|
|40|0.482171|
|41|0.481414|
|42|0.488593|
|43|0.481211|
|44|0.481517|
|45|0.480335|
|46|0.480802|
|47|0.480436|
|48|0.479467|
|49|0.479782|

#### Semantic Segmentation Result

Few images from the semantic segmentation runs. The parts shown in green have been classified as road.

![r1]   
![r2]   
![r3]   
![r4]   
![r5]   
![r6]   


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
