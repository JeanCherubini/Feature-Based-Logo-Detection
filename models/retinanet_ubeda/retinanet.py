# -*- coding: utf-8 -*-
"""This is an TensorFLow implementation of RetinaNet by --- at all.

Paper:
(---)

The pretrained weights can be downloaded here and should be placed in the same
folder as this file:
- ----

@author: 
"""

import tensorflow as tf
import numpy as np
import time
import keras


class RetinaNetPyramid(object):
    
    def __init__(self, x, skip_layer, #keep_prob, num_classes,
             weights_path='DEFAULT',
             include_P2 = False):

        self.X = x
        
        #self.NUM_CLASSES = num_classes
        #self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer
        self.include_P2 = include_P2

        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = '../snapshots/resnet50_coco_best_v2.1.0.h5'
        else:
            self.WEIGHTS_PATH = weights_path
            
        self.create()
    
    def create(self):
        """Create the network graph."""
        start_time = time.time()
        
# =============================================================================
#         tf.reset_default_graph()
#         tf.get_default_graph().get_operations()
#         inputs = tf.placeholder(shape = (50,224,224,3), dtype = tf.float32)                       
# =============================================================================

        self.BOTTOM_UP = ResNet50(self.X, include_top = False)
        C2, C3, C4, C5 = self.BOTTOM_UP
        
        self.TOP_DOWN = create_pyramid_features(C2, C3, C4, C5, feature_size = 256, include_P2 = self.include_P2)
        print(("build model finished: %ds" % (time.time() - start_time)))

        
        self.TOP_DOWN = create_pyramid_features(C2, C3, C4, C5, feature_size = 256, include_P2 = self.include_P2)
        print(("build model finished: %ds" % (time.time() - start_time)))

    def load_initial_weights(self, session, verbose = False):
    
        import h5py
        
        f = h5py.File(self.WEIGHTS_PATH, 'r')
        print(f)
        weights = f['model_weights']
            
        for op_name in sorted(list(weights.keys())):
            if weights.get(op_name).get(op_name) is not None: #only operation with weights
                if verbose:
                    print('Loading weights from: {}'.format(op_name))    
                if op_name not in self.SKIP_LAYER:
                    with tf.compat.v1.variable_scope(op_name, reuse=True):    
                        for data in weights.get(op_name).get(op_name).keys():
                            #get tf var
                            if 'kernel' in data:
                                var = tf.compat.v1.get_variable('weights', trainable=False)
                            elif 'bias'in data:
                                var = tf.compat.v1.get_variable('biases', trainable=False)
                            elif 'beta' in data:
                                var = tf.compat.v1.get_variable('beta', trainable=False)
                            elif 'gamma' in data:
                                var = tf.compat.v1.get_variable('gamma', trainable=False)                                
                            elif 'moving_mean' in data:
                                var = tf.compat.v1.get_variable('moving_mean', trainable=False)                                
                            elif 'moving_variance' in data:
                                var = tf.compat.v1.get_variable('moving_variance', trainable=False)                                                                
                            else:   
                                var = None 
                                
                            #get weights array and assing
                            array = np.array(weights.get(op_name).get(op_name).get(data))
                            session.run(var.assign(array))                                  
        f.close()
        print('Weights file loaded')

def create_pyramid_features(C2, C3, C4, C5, feature_size, include_P2):
    """ Creates the FPN layers on top of the backbone features.
    Args
        C2           : Feature stage C2 from the backbone.
        C3           : Feature stage C3 from the backbone.
        C4           : Feature stage C4 from the backbone.
        C5           : Feature stage C5 from the backbone.
        feature_size : The feature size to use for the resulting feature levels.
        include_P2   : (bool) include P2 level?
    Returns
        A list of feature levels [P2, P3, P4, P5, P6, P7].
    """
    # upsample C5 to get P5 from the FPN paper
    P5 = conv(C5, 1, 1, feature_size, 1, 1, name="C5_reduced", use_bias = True, use_relu = False)
    P5_upsampled = tf.image.resize(P5, size = tf.constant([C4.get_shape().as_list()[1],C4.get_shape().as_list()[2]], dtype = tf.int32), 
                                          method = 1) #method = 1 <=> nearest neighbor (FPN paper)
            
    P5 = conv(P5, 3, 3, feature_size, 1, 1, name="P5", use_bias = True, use_relu = False)
    
    # add P5 elementwise to C4    
    P4 = conv(C4, 1, 1, feature_size, 1, 1, name="C4_reduced", use_bias = True, use_relu = False)
    P4 = tf.add(P5_upsampled, P4, name = 'P4_merged')    
    P4_upsampled = tf.image.resize(P4, size = tf.constant([C3.get_shape().as_list()[1],C3.get_shape().as_list()[2]], dtype = tf.int32), 
                                          method = 1)
    P4 = conv(P4, 3, 3, feature_size, 1, 1, name="P4", use_bias = True, use_relu = False)
    
    # add P4 elementwise to C3
    P3 = conv(C3, 1, 1, feature_size, 1, 1, name="C3_reduced", use_bias = True, use_relu = False)
    P3 = tf.add(P4_upsampled, P3, name = 'P3_merged')
    P3 = conv(P3, 3, 3, feature_size, 1, 1, name="P3", use_bias = True, use_relu = False)

    #C2 is 256 depth so it doesn't need reduction 
    #also we don't have the weights
    P3_upsampled = tf.image.resize(P3, size = tf.constant([C2.get_shape().as_list()[1], C2.get_shape().as_list()[2]], dtype = tf.int32), 
                                          method = 1)
    P2 = tf.add(P3_upsampled, C2, name = 'P2')
    #we don't have weights for the final 3x3 convolution

    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    P6 = conv(C5, 3, 3, feature_size, 2, 2, name="P6", use_bias = True, use_relu = False)
    
    # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    P7 = tf.nn.relu(P6, name='P6_relu')
    P7 = conv(P7, 3, 3, feature_size, 2, 2, name="P7", use_bias = True, use_relu = False)

    if include_P2:
        level_list = [P2, P3, P4, P5, P6, P7]
    else:
        level_list = [P3, P4, P5, P6, P7]
    
    return level_list


def bottleneck_2d(filters, stage=0, block=0, kernel_size=3, axis = -1,
                  numerical_name=False, stride=None, freeze_bn=False):
    """
    A two-dimensional bottleneck block.
    :param filters: the output’s feature space
    :param stage: int representing the stage of this block (starting from 0)
    :param block: int representing this block (starting from 0)
    :param kernel_size: size of the kernel
    :param axis: int representing axis of depth
    :param numerical_name: if true, uses numbers to represent blocks instead of chars (ResNet{101, 152, 200})
    :param stride: int representing the stride used in the shortcut and the first conv layer, default derives stride from block id
    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)
    """
    if stride is None:
        if block != 0 or stage == 0:
            stride = 1
        else:
            stride = 2

    if block > 0 and numerical_name:
        block_char = "b{}".format(block)
    else:
        block_char = chr(ord('a') + block)

    stage_char = str(stage + 2)

    is_training = not freeze_bn

    def f(x):
        
        #1x1        
        y = conv(x, 1, 1, filters, stride, stride, name="res{}{}_branch2a".format(stage_char, block_char), 
             use_bias = False, use_relu = False)
        y = tf.compat.v1.layers.batch_normalization(y, axis=axis, epsilon=1e-5, training = is_training,
                                          name = "bn{}{}_branch2a".format(stage_char, block_char))
        y = tf.nn.relu(y, name="res{}{}_branch2a_relu".format(stage_char, block_char))
        
        #3x3
        #y = tf.keras.layers.ZeroPadding2D(padding=stride, name="padding{}{}_branch2b".format(stage_char, block_char))(y)
        y = conv(y, kernel_size, kernel_size, filters, 1, 1, name="res{}{}_branch2b".format(stage_char, block_char), 
             use_bias = False, use_relu = False)
        y = tf.compat.v1.layers.batch_normalization(y, axis=axis, epsilon=1e-5, training = is_training,
                                          name = "bn{}{}_branch2b".format(stage_char, block_char))
        y = tf.nn.relu(y, name="res{}{}_branch2b_relu".format(stage_char, block_char))
        
        #1x1        
        y = conv(y, 1, 1, filters * 4, 1, 1, name="res{}{}_branch2c".format(stage_char, block_char), 
             use_bias = False, use_relu = False)
        y = tf.compat.v1.layers.batch_normalization(y, axis=axis, epsilon=1e-5, training = is_training,
                                          name = "bn{}{}_branch2c".format(stage_char, block_char))
        
        if block == 0:
            shortcut = conv(x, 1, 1, filters * 4, stride, stride, name="res{}{}_branch1".format(stage_char, block_char), 
                            use_bias = False, use_relu = False)
            shortcut = tf.compat.v1.layers.batch_normalization(shortcut, axis=axis, epsilon=1e-5, training = is_training,
                                          name = "bn{}{}_branch1".format(stage_char, block_char))
        else:
            shortcut = x


        output = tf.add(y, shortcut, name = "res{}{}".format(stage_char, block_char))
        output = tf.nn.relu(output, name="res{}{}_relu".format(stage_char, block_char))
        return output 

    return f


def ResNet(inputs, blocks, block = bottleneck_2d, include_top=True, axis = -1,
           classes=1000, freeze_bn=True, numerical_names=None, *args, **kwargs):
    """
    Constructs a `keras.models.Model` object using the given block count.
    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)
    :param blocks: the network’s residual architecture
    :param block: a residual block (e.g. an instance of `keras_resnet.blocks.basic_2d`)
    :param include_top: if true, includes classification layers
    :param classes: number of classes to classify (include_top must be true)
    :param freeze_bn: if true, freezes BatchNormalization layers (ie. no updates are done in these layers)
    :param numerical_names: list of bool, same size as blocks, used to indicate whether names of layers should include numbers or letters
    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)
    """

    if numerical_names is None:
        numerical_names = [True] * len(blocks)

    is_training = not freeze_bn

    x = conv(inputs, 7, 7, 64, 2, 2, name="conv1", use_bias = False, use_relu = False)
    x = tf.compat.v1.layers.batch_normalization(x, axis=axis, epsilon=1e-5, training = is_training,
                                          name = 'bn_conv1')
    x = tf.nn.relu(x, name="conv1_relu")
    
    x = max_pool(x, 3, 3, 2, 2, padding = 'SAME', name = 'pool1') 
    
    features = 64
    outputs = []

    for stage_id, iterations in enumerate(blocks):
        for block_id in range(iterations):
            x = block(features, stage_id, block_id, numerical_name=(block_id > 0 and numerical_names[stage_id]), freeze_bn=freeze_bn)(x)

        features *= 2

        outputs.append(x)

    if include_top:
        assert classes > 0
        
        #global average pooling
        x = tf.reduce_mean(x, axis = [1,2], name = 'pool5')
        logits = fc(x, 2048, classes, use_relu=False, name='fc1000')
        prob = tf.nn.softmax(logits, name="prob")

        return prob
    else:
        # Else output each stages features
        return outputs

def ResNet50(inputs, blocks=None, include_top=True, classes=1000, *args, **kwargs):
    """
    Constructs a `keras.models.Model` according to the ResNet50 specifications.
    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)
    :param blocks: the network’s residual architecture
    :param include_top: if true, includes classification layers
    :param classes: number of classes to classify (include_top must be true)
    :return model: ResNet model with encoding output (if `include_top=False`) or classification output (if `include_top=True`)
    """

    if blocks is None:
        blocks = [3, 4, 6, 3]
    numerical_names = [False, False, False, False]

    return ResNet(inputs, blocks, numerical_names=numerical_names,
                  include_top=include_top, classes=classes, *args, **kwargs)


##########################################################################################
######################################### LAYERS #########################################
##########################################################################################
                            
def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1, use_bias = True, use_relu = False):
    """Create a convolution layer.

    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.compat.v1.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.compat.v1.get_variable('weights', shape=[filter_height,
                                                    filter_width,
                                                    input_channels/groups,
                                                    num_filters])
        if use_bias:
            biases = tf.compat.v1.get_variable('biases', shape=[num_filters])

    if groups == 1:
        conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(x, groups, 3)
        weight_groups = tf.split(weights, groups, 3)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

        # Concat the convolved output together again
        conv = tf.concat(output_groups, 3 )

    # Add biases
    if use_bias:
        y = tf.nn.bias_add(conv, biases)
    else:
        y = conv
    
    # Apply relu function
    if use_relu:
        output = tf.nn.relu(y, name=scope.name)
    else:
        output = y
    
    return output


def fc(x, num_in, num_out, name, use_relu=True):
    """Create a fully connected layer."""
    with tf.compat.v1.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.compat.v1.get_variable('weights', shape=[num_in, num_out],
                                  trainable=True)
        biases = tf.compat.v1.get_variable('biases', [num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.compat.v1.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if use_relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    else:
        return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
             padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)


def dropout(x, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob)
