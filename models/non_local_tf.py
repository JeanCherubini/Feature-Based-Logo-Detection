
'''

TensorFlow implementation of non-local blocks from AlexHex7/Non-local_pytorch

'''

import tensorflow as tf

#from tf.keras import Sequential

from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D


class NONLocalBlock2D(tf.keras.layers.Layer):
    # use tf.model instead of tf.Module?
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True, **kwargs):
        super(NONLocalBlock2D, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # Assume 2D convolutions, but it can be generalized to other dimensions
        conv_nd = Conv2D
        max_pool_layer = MaxPool2D(pool_size=(2, 2))
        bn = BatchNormalization

        self.g = conv_nd(filters=self.inter_channels, kernel_size=1, strides=1, name='conv_layer_g')

        self.W = tf.keras.Sequential()
        if bn_layer:
            self.W.add(conv_nd(filters=self.in_channels, kernel_size=1, strides=1, name="conv_layer_W"))
            self.W.add(bn(name="bn_layer_W"))
            # set bias/weights of the last layer to 0?
        else:
            self.W.add(conv_nd(filters=self.in_channels, kernel_size=1, strides=1, name="conv_layer_W"))
            # set bias/weights to 0?

        self.theta = conv_nd(filters=self.inter_channels, kernel_size=1, strides=1, name='conv_layer_theta')
        self.phi = conv_nd(filters=self.inter_channels, kernel_size=1, strides=1, name='conv_layer_phi')
        
        # add sub_sample option to reduce computation?
        # if sub_sample:
        #     self.g = tf.keras.Sequential(self.g, max_pool_layer) 
        #     self.phi = tf.keras.Sequential(self.phi, max_pool_layer)


        
    def __call__(self, x, return_nl_map=False):
        """
        :param x: (b, h, w, c)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """
        #print("input.shape = {}".format(x.shape))
        #print("g(x).shape = {}".format(self.g(x).shape))

        #batch_size = x.shape[0]
        #batch_size = x.get_shape()[0]
        #print('batch_size', batch_size)
        height = x.shape[1]
        width = x.shape[2]


        g_x = tf.reshape(self.g(x), [-1, height*width, self.inter_channels])

        theta_x = tf.reshape(self.theta(x), [-1, height*width, self.inter_channels])
        phi_x = tf.reshape(self.phi(x), [-1, height*width, self.inter_channels])
        phi_x = tf.keras.backend.permute_dimensions(phi_x, (0, 2, 1))
        
        f = tf.linalg.matmul(theta_x, phi_x)
        N = f.shape[-1]
        f_div_C = f / N
        
        y = tf.linalg.matmul(f_div_C, g_x)
        y = tf.reshape(y, [-1, x.shape[1], x.shape[2], self.inter_channels])
        W_y = self.W(y)
        z = W_y + x

        # add Squeeze and Excitation?
        
        if return_nl_map:
            return z, f_div_C
        return z



if __name__ == '__main__':

    for (sub_sample_, bn_layer_) in [(True, True), (False, False), (True, False), (False, True)]:
        print("\nsub_sample_ = {}, bn_layer_ = {}".format(sub_sample_, bn_layer_))
        img = tf.zeros([2, 20, 20, 3])
        net = NONLocalBlock2D(3, sub_sample=sub_sample_, bn_layer=bn_layer_)
        out = net(img)
        print(out.shape)
