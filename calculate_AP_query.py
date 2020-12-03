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
            #print(img_id, annot)
            #plt.imshow(train_images.load_image(img_id)/255)
            #plt.show()

    #check if the detections made are true positives or false positives according to the iou threshhold
    for img_id in detections.keys():
        #image of the detection exists in the ground truth
        if(img_id in all_annotations_this_class.keys()):
            already_found = []
            #if the detection exists in the image, check IoU over all of the annotations in the image
            for det in detections[img_id]:
                for annot in all_annotations_this_class[img_id]:
                    IoU = bb_intersection_over_union(det, annot)
                    #If detection is sufficient we add a true detection and discount a false negative detection
                    if IoU>=th_IoU:
                        #Check if instance was already found
                        if (str(annot) not in already_found):
                            true_positives+=1
                            false_negatives-=1
                            #save detection for non repetition
                            already_found.append(str(annot))
                        else:
                            print('repetition')
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

    print('TP:', true_positives)
    print('FP:', false_positives)
    print('FN:', false_negatives)
    
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
        precisions.append(0)
        recalls.append(recalls[-1]+1e-9)

    plt.plot(recalls, precisions_smoothed, '-g*')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()
    return recalls, precisions_smoothed

    def calculate_interpolated_AP(recalls, precision):
        ranges = np.arange(0,1,0.1)
        maximum_in_range = [1]

        current_range = 1
        for r in range(len(recalls)):
            current_max = 0
            if(recalls[r] <= precision[0]):
                print()

        print(ranges)
        

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
            try:
                detections[id_].append(bbox)
                detection_values[id_].append(value)
            except:
                detections[id_]=[bbox]
                detection_values[id_]=[value]
                continue
    print('detections', detections)
    print('detection_values', detection_values)

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

    #calculate precision recall
    recalls, precisions = calculate_precision_recall(detections, all_annotations_this_class, 0.5)
    #calculate_interpolated_AP = calculate_interpolated_AP(recalls, precisions)


    top = 1

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
