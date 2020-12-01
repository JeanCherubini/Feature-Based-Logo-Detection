import sys
import tensorflow as tf
from models import resnet#, uv_rois
import numpy as np
import argparse
import os
import random
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
import pickle as pk
from datetime import datetime

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils.COCO_Utils.COCO_like_dataset import CocoLikeDataset 

#Fuction modified from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[0] + boxA[2]-1, boxB[0] + boxB[2]-1)
	yB = min(boxA[1] + boxA[3]-1, boxB[1] + boxB[3]-1)
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2]) * (boxA[3])
	boxBArea = (boxB[2]) * (boxB[3])
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset_name', help='dataset name', type=str, choices=['DocExplore', 'flickrlogos_47'], default='flickrlogos_47')
    parser.add_argument('-coco_images', help='image directory in coco format', type=str, default = '/mnt/BE6CA2E26CA294A5/Datasets/flickrlogos_47_COCO/images/train')
    parser.add_argument('-annotation_json', help='image directory in coco format', type=str, default = '/mnt/BE6CA2E26CA294A5/Datasets/flickrlogos_47_COCO/annotations/instances_train.json')
    parser.add_argument('-query_path', help='path to queries', type=str, default = '/mnt/BE6CA2E26CA294A5/Datasets/flickrlogos_47_COCO/images/queries_train/')
    parser.add_argument('-query_class', help='class of the query', type=str, default = 'adidas_symbol')
    parser.add_argument('-query_instance', help = 'filename of the query', type=str, default = 'random')
    parser.add_argument('-feat_savedir', help='directory of features database', type=str, default='/home/jeancherubini/Documents/feature_maps')
    parser.add_argument('-principal_components', help='amount of components kept (depth of feature vectors)', type=int, default=64)
    parser.add_argument('-model', help='model used for the convolutional features', type=str, choices=['resnet', 'VGG16'], default='VGG16') 
    parser.add_argument('-layer', help='resnet layer used for extraction', type=str, choices=['conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out', 'block3_conv3', 'block4_conv3', 'block5_conv3'], default='block3_conv3') 
    parser.add_argument('-p', help='max points collected from each heatmap', type=int, default=15)

    params = parser.parse_args()    

    #Model correction features map
    params.feat_savedir = params.feat_savedir+'/'+params.dataset_name

    #creation of dataset like coco
    train_images = CocoLikeDataset()
    train_images.load_data(params.annotation_json, params.coco_images)
    train_images.prepare()

    classes_dictionary = train_images.class_info
    query_class_num = [cat['id'] for cat in classes_dictionary if cat['name']==params.query_class][0]

    #load desired query results
    query_results = open('{0}/results/{1}/{2}.txt'.format(params.feat_savedir, params.query_class,params.query_instance.replace('.png','')), 'r')

    #get all detections for each image
    detections = {}

    for row in query_results:
        id_ = int(row.replace('.png', '').split(' ')[0])
        bbox = row.replace('.png', '').split(' ')[1:5]
        bbox = [int(coord) for coord in bbox]
        try:
            detections[id_].append(bbox)
        except:
            detections[id_]=[bbox]
            continue
    print('detections', detections)

    #get all ground truth annotations for the class of the query
    all_annotations_this_class = {}

    all_image_ids = train_images.image_ids
    for image_id in all_image_ids:
        annotations_this_image = train_images.load_annotations(image_id)
        this_class_annotations = []
        for annot in annotations_this_image:
            if(annot[-1]==query_class_num):
                this_class_annotations.append(annot[:-1])
        if(this_class_annotations):
            all_annotations_this_class[image_id] = this_class_annotations
    
    print('ground truth', all_annotations_this_class)

    #assertions for IoU
    assert(bb_intersection_over_union([0, 0, 5, 5], [0, 0, 10, 10])==0.25)
    assert(bb_intersection_over_union([0, 0, 5, 10], [0, 0, 10, 10])==0.5)
    assert(bb_intersection_over_union([0, 0, 10, 10], [0, 0, 10, 10])==1.0)
    assert(bb_intersection_over_union([0, 0, 10, 10], [10, 10, 10, 10])==0.0)

    
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    #check false negatives, every ground truth that is not in the detections made
    for img_id in all_annotations_this_class.keys():
        if(img_id not in detections.keys()):
            #add one false negative for each wrong detection
            for annot in all_annotations_this_class[img_id]:
                false_negatives += 1
                print(annot)

    #check if the detections made are true positives or false positives according to the iou threshhold
    for img_id in detections.keys():
        if(img_id in all_annotations_this_class.keys()):
            #if the detection exists in the image, check IoU over all of the annotations in the image
            for det in detections[img_id]:
                for annot in all_annotations_this_class[img_id]:
                    IoU = bb_intersection_over_union(det, annot)
                    if IoU>=0.5:
                        true_positives+=1
                    else:
                        false_positives+=1
        else:
            false_positives+=1

    precision = true_positives/(true_positives+false_positives)
    recall = true_positives/(true_positives+false_negatives)
    print(precision, recall)
    
