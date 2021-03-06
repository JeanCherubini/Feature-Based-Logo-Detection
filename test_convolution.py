import tensorflow as tf
import numpy as np

image = tf.constant(range(12), shape=(1,2,2,3), dtype=tf.float32)
print(image.shape)
print(image[:,:,:,0])


query = tf.constant(range(12), shape=(2,2,3), dtype=tf.float32)

print(query.shape)
print(query[:,:,0])

query_reshaped = tf.expand_dims(query,-1)
print(query_reshaped.shape)

conv = tf.nn.conv2d(image,query_reshaped, padding = 'SAME', strides=(1,1,1,1))

print(conv.shape)
print(conv)