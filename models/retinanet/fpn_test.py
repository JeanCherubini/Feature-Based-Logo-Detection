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
from retinanet import *


"""
## Downloading the COCO2017 dataset

Training on the entire COCO2017 dataset which has around 118k images takes a
lot of time, hence we will be using a smaller subset of ~500 images for
training in this example.
"""

url = "https://github.com/srihari-humbarwadi/datasets/releases/download/v0.1.0/data.zip"
filename = os.path.join('/mnt/BE6CA2E26CA294A5/Datasets/', "data.zip")
keras.utils.get_file(filename, url)


with zipfile.ZipFile("/mnt/BE6CA2E26CA294A5/Datasets/data.zip", "r") as z_fp:
    z_fp.extractall("/mnt/BE6CA2E26CA294A5/COCO_2017")

"""
## Implementing utility functions

Bounding boxes can be represented in multiple ways, the most common formats are:

- Storing the coordinates of the corners `[xmin, ymin, xmax, ymax]`
- Storing the coordinates of the center and the box dimensions
`[x, y, width, height]`

Since we require both formats, we will be implementing functions for converting
between the formats.
"""




"""
## Setting up training parameters
"""

model_dir = "retinanet/"
label_encoder = LabelEncoder()

num_classes = 80
batch_size = 1

"""
## Initializing and compiling model
"""

resnet50_backbone = get_backbone()
model = FPN_model(num_classes, resnet50_backbone)



"""
## Loading weights
"""

# Change this to `model_dir` when not using the downloaded weights
weights_dir = "/mnt/BE6CA2E26CA294A5/Datasets/COCO_2017/data"

latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
model.load_weights(latest_checkpoint)


"""
## Load the COCO2017 dataset using TensorFlow Datasets
"""

#  set `data_dir=None` to load the complete dataset

(train_dataset, val_dataset), dataset_info = tfds.load(
    "coco/2017", split=["train", "validation"], with_info=True, data_dir="/mnt/BE6CA2E26CA294A5/Datasets/COCO_2017/data"
)

"""
## Building inference model
"""

image = tf.keras.Input(shape=[None, None, 3], name="image")

features = model(image, training=False)

inference_model = tf.keras.Model(inputs=image, outputs=features)



#print(intermediate_model.summary())


val_dataset = tfds.load("coco/2017", split="validation", data_dir="/mnt/BE6CA2E26CA294A5/Datasets/COCO_2017")
int2str = dataset_info.features["objects"]["label"].int2str



for sample in val_dataset.take(2):
    image = tf.cast(sample["image"], dtype=tf.float32)
    plt.imshow(image[0])
    plt.figure()
    input_image, ratio = prepare_image(image)
    plt.imshow(input_image[0])
    plt.figure()
    features = inference_model.predict(input_image)
    plt.imshow(features[0][0,:,:,0])
    plt.show()
    for i in features:
        print(i.shape)
 

