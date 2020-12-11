import os

import argparse

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset_name', help='dataset name', type=str, choices=['DocExplore', 'flickrlogos_47'], default='DocExplore')
    parser.add_argument('-model', help='model used for the convolutional features', type=str, choices=['resnet', 'VGG16'], default='VGG16') 
    parser.add_argument('-layer', help='resnet layer used for extraction', type=str, choices=['conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out', 'block3_conv3', 'block4_conv3', 'block5_conv3'], default='block3_conv3') 
    parser.add_argument('-feat_savedir', help='directory of features database', type=str, default='/home/jeancherubini/Documents/feature_maps')

    params = parser.parse_args()    

    
    #Open file with detections ordered by value
    all_detections_ordered = open('{0}/{1}/{2}/detections/all_detections_ordered.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer),'r')

    detections_by_query_id = {}


    for row in all_detections_ordered:
        #image retrieval
        query_id, image_detected, x1, y1, width, height, value, query_class = row.split(' ')
        if query_id not in detections_by_query_id.keys():
            detections_by_query_id[query_id]=[query_id, image_detected, x1, y1, width, height, value, query_class]


    #open 
    print(detections_by_query_id)
