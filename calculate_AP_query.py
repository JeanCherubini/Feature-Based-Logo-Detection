import sys
import tensorflow as tf

import numpy as np
import argparse
import os
import random
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
import pickle as pk
from datetime import datetime
from matplotlib.patches import Rectangle

import collections

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils.COCO_Utils.COCO_like_dataset import CocoLikeDataset 
    
from calculate_AP_class import *

import json

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    #parser.add_argument('-dataset_name', help='dataset name', type=str, choices=['DocExplore', 'flickrlogos_47'], default='flickrlogos_47')
    #parser.add_argument('-coco_images', help='image directory in coco format', type=str, default = '/mnt/BE6CA2E26CA294A5/Datasets/flickrlogos_47_COCO/images/train')
    #parser.add_argument('-annotation_json', help='image directory in coco format', type=str, default = '/mnt/BE6CA2E26CA294A5/Datasets/flickrlogos_47_COCO/annotations/instances_train.json')
    #parser.add_argument('-query_path', help='path to queries', type=str, default = '/mnt/BE6CA2E26CA294A5/Datasets/flickrlogos_47_COCO/images/queries_train/')
    parser.add_argument('-model', help='model used for the convolutional features', type=str, choices=['resnet', 'VGG16'], default='VGG16') 
    parser.add_argument('-layer', help='resnet layer(s) used for extraction, they can be:\n for VGG: {0}\n for resnet:{1}\n For multiple layers, a semicolon "," can be used to separate '.format(
    'conv1_relu, conv2_block3_out, conv3_block4_out, conv4_block6_out, conv5_block3_out',
    'block3_conv3, block4_conv3, block5_conv3'), type=str, default='block3_conv3') 
    parser.add_argument('-query_class', help='class of the query', type=str, default = 'adidas_symbol')
    parser.add_argument('-query_instance', help = 'filename of the query', type=str, default = 'random')
    #parser.add_argument('-feat_savedir', help='directory of features database', type=str, default='/home/jeancherubini/Documents/feature_maps')
    parser.add_argument('-principal_components', help='amount of components kept (depth of feature vectors)', type=str, default='64')   
    parser.add_argument('-th_value', help='threshhold value to keep image', type=float, default=0.1)
    parser.add_argument('-cfg', help='config file with paths', type=str)


    params = parser.parse_args()   
    
    #Complete argswith routes from config file
    with open(params.cfg) as json_data_file:
        cfg_data = json.load(json_data_file)
    
    params.dataset_name = cfg_data['dataset_name']
    params.coco_images = cfg_data['coco_images']
    params.annotation_json = cfg_data['annotation_json'] 
    params.query_path = cfg_data['query_path']
    params.feat_savedir = cfg_data['feat_savedir']

    AP_calculator = AP_calculator_class()

   
    #create ordered detections file
    AP_calculator.get_ordered_detections(params, params.query_class, params.query_instance)
    
    AP_calculator.plt_top_detections(params, params.query_class, params.query_instance)
    AP_calculator.calculate_query_ps(params, params.query_class, params.query_instance)