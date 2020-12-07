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

def calculate_precision_recall(detections, all_annotations_this_class, th_IoU):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    precisions = [1]
    recalls = [0]


    #false negatives start as the amount of instances in the set
    for img_id in all_annotations_this_class.keys():
        for annot in all_annotations_this_class[img_id]:
            false_negatives += 1


    already_found = []
    #check if the detections made are true positives or false positives according to the iou threshhold
    for value, img_id in detections.keys():
        #image of the detection exists in the ground truth
        if(img_id in all_annotations_this_class.keys()):
            #if the detection exists in the image, check IoU over all of the annotations in the image
            bbox_detection = detections[value, img_id]
            for annot in all_annotations_this_class[img_id]:
                IoU = bb_intersection_over_union(bbox_detection, annot)
                #If detection is sufficient we add a true detection and discount a false negative detection, if this annotations wasnt already found
                if IoU>=th_IoU:
                    #Check if instance was already found
                    if (str(img_id)+':'+str(annot) not in already_found):
                        true_positives+=1
                        false_negatives-=1
                        #save detection for non repetition
                        already_found.append(str(img_id)+':'+str(annot))
                    else:
                        false_positives+=1



                #If detection is not enough
                else:
                    false_positives+=1
        #image of the detection is not in the ground truth, thus it does not contain the query at all
        else:
            false_positives+=1

        try:
            precisions.append(true_positives/(true_positives+false_positives))
            recalls.append(true_positives/(true_positives+false_negatives))
        except:
            return

    
    #smoothing of precisions
    precisions_smoothed = precisions
    for i in range(len(precisions)-2,-1,-1):
        max_to_right = np.max(precisions[i:])
        if precisions_smoothed[i]<max_to_right:
            precisions_smoothed[i]=max_to_right
        else:
            precisions_smoothed[i]=precisions[i]
    #end curve
    if recalls[-1]<1:
        precisions.append(1e-15)
        recalls.append(recalls[-1]+1e-15)

    plt.plot(recalls, precisions_smoothed, '-gp')
    plt.xlim(0,1)
    plt.ylim(0,1)
    return np.array(recalls), np.array(precisions_smoothed)

def calculate_interpolated_AP(recalls, precision, delta):

    maximums_in_range = []
    minimums_in_range = []
    ready = 0

    for r in np.arange(0,1,delta):
        values_that_exist_this_range = (recalls>=r)&(recalls<r+delta)
        valid_indexes = np.where(values_that_exist_this_range)[0]
        values = precision[valid_indexes]
        
        if(ready==0):
            if(len(values)!=0):
                max_in_range = np.max(values)
                maximums_in_range.append(max_in_range)

                min_in_range = np.min(values)
                minimums_in_range.append(min_in_range)

                if(1e-15 in precision[valid_indexes]):
                    ready = 1
            else:
                if(maximums_in_range[-1]>minimums_in_range[-1]):
                    maximums_in_range.append(minimums_in_range[-1])
                else:
                    maximums_in_range.append(maximums_in_range[-1])

        else:
            max_in_range = 0
            maximums_in_range.append(0)


    AP = 100*np.sum(np.array(maximums_in_range))/len(maximums_in_range)
    plt.plot(np.arange(0,1,delta), maximums_in_range, '*r')
    plt.plot(np.arange(0,1,delta), np.zeros_like(np.arange(0,1,delta)), 'ob')
    

    plt.xlim(0,1)
    plt.ylim(0,1)
    display=0
    if display:
        plt.show(block=False)
        plt.pause(2)
        plt.close()
    return AP

    

    

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset_name', help='dataset name', type=str, choices=['DocExplore', 'flickrlogos_47'], default='flickrlogos_47')
    parser.add_argument('-coco_images', help='image directory in coco format', type=str, default = '/mnt/BE6CA2E26CA294A5/Datasets/flickrlogos_47_COCO/images/train')
    parser.add_argument('-annotation_json', help='image directory in coco format', type=str, default = '/mnt/BE6CA2E26CA294A5/Datasets/flickrlogos_47_COCO/annotations/instances_train.json')
    parser.add_argument('-query_path', help='path to queries', type=str, default = '/mnt/BE6CA2E26CA294A5/Datasets/flickrlogos_47_COCO/images/queries_train/')
    parser.add_argument('-query_class', help='class of the query', type=str, default = 'adidas_symbol')
    parser.add_argument('-query_instance', help = 'filename of the query', type=str, default = 'random')
    parser.add_argument('-feat_savedir', help='directory of features database', type=str, default='/home/jeancherubini/Documents/feature_maps')
    parser.add_argument('-th_value', help='threshhold value to keep image', type=float, default=0.5)

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
    query_results = open('{0}/detections/{1}/{2}.txt'.format(params.feat_savedir, params.query_class,params.query_instance.replace('.png','')), 'r')

    #get all detections for each image
    detections = {}
    detection_values = {}

    for row in query_results:
        id_ = int(row.split(' ')[0])
        bbox = row.split(' ')[1:5]
        bbox = [int(coord) for coord in bbox]
        value = float(row.split(' ')[-2])
        if value>=params.th_value:
            detections[value, id_]=bbox
    

    ordered_detections = collections.OrderedDict(sorted(detections.items(), reverse=True))

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
    

    #assertions for IoU
    assert(bb_intersection_over_union([0, 0, 5, 5], [0, 0, 10, 10])==0.25)
    assert(bb_intersection_over_union([0, 0, 5, 10], [0, 0, 10, 10])==0.5)
    assert(bb_intersection_over_union([0, 0, 10, 10], [0, 0, 10, 10])==1.0)
    assert(bb_intersection_over_union([0, 0, 10, 10], [10, 10, 10, 10])==0.0)

    multiple_ious = [0.05 , 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    APS = {}


    if not os.path.isdir(params.feat_savedir + '/AP'):
        os.mkdir(params.feat_savedir + '/AP')
    
    if not os.path.isdir(params.feat_savedir + '/AP/' + params.query_class):
        os.mkdir(params.feat_savedir + '/AP/' + params.query_class)
    
    file_AP = open('{0}/AP/{1}/{2}.txt'.format(params.feat_savedir, params.query_class, params.query_instance.replace('.png', '')), 'w')


    for iou in multiple_ious:
        #calculate precision recall
        recalls, precisions = calculate_precision_recall(ordered_detections, all_annotations_this_class, iou)
        calculated_interpolated_AP = calculate_interpolated_AP(recalls, precisions,0.01)
        file_AP.write('{0}:{1:2.2f}\t'.format(iou, calculated_interpolated_AP) )
        APS[iou] = calculated_interpolated_AP
    print(params.query_instance, APS)

    
    file_AP.close()


    top = 0

    if top:
        #create figure to show query
        #plt.figure()
        #plt.imshow(query)
        if not os.path.isdir(params.feat_savedir + '/results'):
            os.mkdir(params.feat_savedir + '/results')
        
        for i,id_ in enumerate(detections.keys()):
            n=i%10
            if n==0:
                if i!=0:
                    plt.savefig('{0}/results/{1}_{2}_top_{3}'.format(params.feat_savedir, params.query_class, str(i), params.query_instance))
                    plt.show(block=False)
                    plt.pause(3)
                    plt.close()
                fig, ([ax0, ax1, ax2, ax3, ax4], [ax5, ax6, ax7, ax8, ax9]) = plt.subplots(2, 5, sharey=False, figsize=(25,15))
                axs = ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9 
            

            #image load
            image = train_images.load_image(id_)
            axs[n].imshow(image)
            

            #get detections for this image
            bounding_box = detections[id_]

            for bbox in bounding_box:
                x1, y1, width, height = bbox
                if not ([x1, y1, width, height]==[0 ,0 , 0 ,0]):
                    rect = Rectangle((x1,y1), width, height, edgecolor='r', facecolor="none")
                    axs[n].add_patch(rect)
            try:
                #get ground truth for this image
                annotation = train_images.load_annotations(id_)
                for ann in annotation:
                    x1, y1 ,width ,height, label = ann 
                    if not ([x1, y1, width, height]==[0 ,0 , 0 ,0]):
                        if(int(query_class_num)==int(label)):         
                            rect = Rectangle((x1,y1), width, height, edgecolor='g', facecolor="none")
                            axs[n].add_patch(rect)
            except:
                continue
            

        
        plt.savefig('{0}/results/{1}_{2}_top_{3}'.format(params.feat_savedir, params.query_class, 'last', params.query_instance))
        plt.show(block=False)
        plt.pause(3)
        plt.close()
