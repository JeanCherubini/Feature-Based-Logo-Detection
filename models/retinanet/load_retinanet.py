"""
Title: Object Detection with RetinaNet
Author: [Srihari Humbarwadi](https://twitter.com/srihari_rh)
Date created: 2020/05/17
Last modified: 2020/07/14
Description: Implementing RetinaNet: Focal Loss for Dense Object Detection.
"""

"""
## Introduction

Object detection a very important problem in computer
vision. Here the model is tasked with localizing the objects present in an
image, and at the same time, classifying them into different categories.
Object detection models can be broadly classified into "single-stage" and
"two-stage" detectors. Two-stage detectors are often more accurate but at the
cost of being slower. Here in this example, we will implement RetinaNet,
a popular single-stage detector, which is accurate and runs fast.
RetinaNet uses a feature pyramid network to efficiently detect objects at
multiple scales and introduces a new loss, the Focal loss function, to alleviate
the problem of the extreme foreground-background class imbalance.

**References:**

- [RetinaNet Paper](https://arxiv.org/abs/1708.02002)
- [Feature Pyramid Network Paper](https://arxiv.org/abs/1612.03144)
"""


import os
import re
import zipfile

import numpy as np
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from . import retinanet

def load_retinanet_FPN_model():

    num_classes = 80
    batch_size = 1


    """
    ## Initializing and compiling model
    """
    resnet50_backbone = retinanet.get_backbone()

    model = retinanet.FPN_model(num_classes, resnet50_backbone)


    # Change this to `model_dir` when not using the downloaded weights
    weights_dir = "/mnt/BE6CA2E26CA294A5/Datasets/COCO_2017/data"

    latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
    model.load_weights(latest_checkpoint)

    """
    ## Building inference model
    """

    image = tf.keras.Input(shape=[None, None, 3], name="image")
    features = model(image, training=False)
    model = tf.keras.Model(inputs=image, outputs=features)
    return model

if __name__ == "__main__":
    a = load_retinanet_FPN_model()
    print(a.summary())