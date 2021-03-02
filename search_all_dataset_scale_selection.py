
import sys
import tensorflow as tf
import numpy as np
import argparse
import os
import random
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from collections import defaultdict

from sklearn.decomposition import PCA

import pickle as pk
from datetime import datetime

from skimage.io import imread

from PIL import Image, ImageDraw 

from time import time

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils.COCO_Utils.COCO_like_dataset import CocoLikeDataset 

from query_finder_class import * 

import json



if __name__ == '__main__' :

    # GPU OPTIONS
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    parser = argparse.ArgumentParser()
    #parser.add_argument('-dataset_name', help='dataset name', type=str, choices=['DocExplore', 'flickrlogos_47'], default='flickrlogos_47')
    #parser.add_argument('-coco_images', help='image directory in coco format', type=str, default = '/mnt/BE6CA2E26CA294A5/Datasets/flickrlogos_47_COCO/images/train')
    #parser.add_argument('-annotation_json', help='image directory in coco format', type=str, default = '/mnt/BE6CA2E26CA294A5/Datasets/flickrlogos_47_COCO/annotations/instances_train.json')
    #parser.add_argument('-query_path', help='path to queries', type=str, default = '/mnt/BE6CA2E26CA294A5/Datasets/flickrlogos_47_COCO/images/queries_train/')
    parser.add_argument('-query_class', help='class of the query', type=str, default = 'adidas_symbol')
    parser.add_argument('-query_instance', help = 'filename of the query', type=str, default = 'random')
    #parser.add_argument('-feat_savedir', help='directory of features database', type=str, default='/home/jeancherubini/Documents/feature_maps')
    parser.add_argument('-principal_components', help='amount of components kept (depth of feature vectors)', type=int, default=64)
    parser.add_argument('-model', help='model used for the convolutional features', type=str, choices=['resnet', 'VGG16'], default='VGG16') 
    parser.add_argument('-layer', help='resnet layer(s) used for extraction, they can be:\n for VGG: {0}\n for resnet:{1}\n For multiple layers, a semicolon "," can be used to separate '.format(
    'conv1_relu, conv2_block3_out, conv3_block4_out, conv4_block6_out, conv5_block3_out',
    'block1_conv2, block2_conv2, block3_conv3, block4_conv3, block5_conv3'), type=str, default='block3_conv3') 
    parser.add_argument('-p', help='max points collected from each heatmap', type=int, default=15)
    parser.add_argument('-cfg', help='config file with paths', type=str)
    parser.add_argument('-all', help='search all dataset queries or only one of them per class', type=int, default=0)

    
    params = parser.parse_args()

    #Complete argswith routes from config file
    with open(params.cfg) as json_data_file:
        cfg_data = json.load(json_data_file)
    
    params.dataset_name = cfg_data['dataset_name']
    params.coco_images = cfg_data['coco_images']
    params.annotation_json = cfg_data['annotation_json'] 
    params.query_path = cfg_data['query_path']
    params.feat_savedir = cfg_data['feat_savedir']


    finder = query_finder()

    if not os.path.isdir(params.feat_savedir + '/' + params.dataset_name + '/' + params.model + '_scale_selection/'):
                os.mkdir(params.feat_savedir +'/' + params.dataset_name + '/' + params.model + '_scale_selection/')

    if not os.path.isdir(params.feat_savedir + '/' + params.dataset_name + '/' + params.model + '_scale_selection/' + str(params.principal_components)):
        os.mkdir(params.feat_savedir +'/' + params.dataset_name + '/' + params.model + '_scale_selection/' + str(params.principal_components))

    if not os.path.isdir(params.feat_savedir + '/' + params.dataset_name + '/' + params.model + '_scale_selection/' + str(params.principal_components) + '/detections'):
        os.mkdir(params.feat_savedir +'/' + params.dataset_name + '/' + params.model + '_scale_selection/' + str(params.principal_components) + '/detections')

    
    time_file_scale_selection = open('{0}/{1}/{2}/{3}/detections/time.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_scale_selection', params.principal_components),'w')
    time_file_scale_selection.close()

    for query_class in os.listdir(params.query_path):
        for query_instance in sorted(os.listdir(params.query_path + '/' + query_class)):
            try:
                query = finder.get_query(params, query_class, query_instance)
                layer_to_use = finder.select_scale_query(params, query)
                params.layer = layer_to_use
                queries_transformated = finder.get_query_transformations(query)
                finder.search_query(params, query_class, query_instance, queries_transformated)

                #get detection results to resume in one folder
                results_query = open('{0}/{1}/{2}/{3}/detections/{4}/{5}.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer, params.principal_components,  query_class, query_instance.replace('.png','').replace('.jpg','')),'r')
                
                

                if not os.path.isdir(params.feat_savedir + '/' + params.dataset_name + '/' + params.model + '_scale_selection/' + str(params.principal_components) + '/detections/'+query_class):
                    os.mkdir(params.feat_savedir + '/' + params.dataset_name + '/' + params.model + '_scale_selection/' + str(params.principal_components) + '/detections/'+query_class)

                results_scale_selection = open('{0}/{1}/{2}/{3}/detections/{4}/{5}.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_scale_selection', params.principal_components,  query_class, query_instance.replace('.png','').replace('.jpg','')),'w')
                for line in results_query.readlines():
                    results_scale_selection.write(line)

                #Query times
                times_file = open('{0}/{1}/{2}/{3}/detections/time.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer, params.principal_components),'r')
                time_file_scale_selection = open('{0}/{1}/{2}/{3}/detections/time.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_scale_selection', params.principal_components),'a')


                for line in times_file.readlines():
                    instance = line.split('\t')[0]
                    if instance == query_instance:
                        time_file_scale_selection.write(params.layer + '\t' + line)


                time_file_scale_selection.close()
            except:
                continue
            if not params.all:
                break
    

            
