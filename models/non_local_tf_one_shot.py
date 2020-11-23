'''

TensorFlow implementation of non-local blocks from timy90022/One-Shot-Object-Detection/lib/model/faster_rcnn

'''

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D

class match_block(tf.keras.layers.Layer):
    # use tf.model instead of tf.Module?
    def __init__(self, inplanes):
        super(match_block, self).__init__()

        self.in_channels = inplanes

        self.inter_channels = self.in_channels // 2
        
        if self.inter_channels == 0:
            self.inter_channels = 1

        # Assume 2D convolutions, but it can be generalized to other dimensions
        conv_nd = Conv2D
        max_pool_layer = MaxPool2D(pool_size=(2, 2))
        bn = BatchNormalization

        self.g = conv_nd(filters=self.inter_channels, kernel_size=1, strides=1)

        self.W = tf.keras.Sequential()
        self.W.add(conv_nd(filters=self.in_channels, kernel_size=1, strides=1, name="conv_layer"))
        self.W.add(bn(name="bn_layer"))  # Add argument to enable/disable batch normalization?
        # Set last layer weight and bias to 0

        self.Q = tf.keras.Sequential()
        self.Q.add(conv_nd(filters=self.in_channels, kernel_size=1, strides=1, name="conv_layer"))
        self.Q.add(bn(name="bn_layer"))  # Add option to enable/disable batch normalization?
        # Set last layer weight and bias to 0

        self.theta = conv_nd(filters=self.inter_channels, kernel_size=1, strides=1)
        self.phi = conv_nd(filters=self.inter_channels, kernel_size=1, strides=1)

        # add sub_sample option to reduce computation?
        # if sub_sample:
        #     self.g = tf.keras.Sequential(self.g, max_pool_layer)
        #     self.phi = tf.keras.Sequential(self.phi, max_pool_layer)



    def __call__(self, detect, aim):
        """
        :param detect: (b, h, w, c)
        :param aim: (b, h, w, c)
        :return:
        """
        batch_size, height_a, width_a, channels = aim.shape      # query
        batch_size, height_d, width_d, channels = detect.shape   # target

        #### find aim image similar object ####
        d_x = tf.reshape(self.g(detect), [-1, height_d*width_d, self.inter_channels])
        a_x = tf.reshape(self.g(aim), [-1, height_a*width_a, self.inter_channels])

        theta_x = tf.reshape(self.theta(aim), [-1, height_a*width_a, self.inter_channels])

        phi_x = tf.reshape(self.phi(detect), [-1, height_d*width_d, self.inter_channels])
        phi_x = tf.keras.backend.permute_dimensions(phi_x, (0, 2, 1))

        f = tf.linalg.matmul(theta_x, phi_x)
        N = f.shape[-1]
        f_div_C = f / N

        f = tf.keras.backend.permute_dimensions(f, (0, 2, 1))
        N = f.shape[-1]
        fi_div_C = f / N

        non_aim = tf.linalg.matmul(f_div_C, d_x)
        non_aim = tf.reshape(non_aim, [-1, height_a, width_a, self.inter_channels])
        non_aim = self.W(non_aim)
        non_aim = non_aim + aim

        non_det = tf.linalg.matmul(fi_div_C, a_x)
        non_det = tf.reshape(non_det, [-1, height_d, width_d, self.inter_channels])
        non_det = self.Q(non_det)
        non_det = non_det + detect


        # add Squeeze and Excitation?

        return non_aim, non_det


if __name__ == '__main__':

    detect_feat = tf.zeros([2, 20, 20, 16])
    query_feat = tf.zeros([2, 5, 5, 16])

    net = match_block(inplanes=16)
    non_aim, non_det = net(detect_feat, query_feat)

    print("detect_feat.shape = {}".format(detect_feat.shape))
    print("query_feat.shape = {}".format(query_feat.shape))
    print("non_aim.shape = {}".format(non_aim.shape))
    print("non_det.shape = {}".format(non_det.shape))
