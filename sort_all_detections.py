import os

import argparse

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset_name', help='dataset name', type=str, choices=['DocExplore', 'flickrlogos_47'], default='flickrlogos_47')
    parser.add_argument('-model', help='model used for the convolutional features', type=str, choices=['resnet', 'VGG16'], default='VGG16') 
    parser.add_argument('-layer', help='resnet layer used for extraction', type=str, choices=['conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out', 'block3_conv3', 'block4_conv3', 'block5_conv3'], default='block3_conv3') 
    parser.add_argument('-feat_savedir', help='directory of features database', type=str, default='/home/jeancherubini/Documents/feature_maps')
    parser.add_argument('-th_value', help='threshhold value to keep image', type=float, default=0.1)

    params = parser.parse_args()    

    

    #Open all detections document
    all_detections = open('{0}/{1}/{2}/detections/all_detections.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer),'r')
    
    #Open file with detections ordered by value
    all_detections_ordered = open('{0}/{1}/{2}/detections/all_detections_ordered.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer),'w')

    detections_by_value_and_query_id = {}


    for row in all_detections:
        #image retrieval
        query_id, image_detected, x1, y1, width, height, value, query_class = row.split(' ') 
        detections_by_value_and_query_id[float(value), query_id]=[query_id, image_detected, x1, y1, width, height, value, query_class]

    ordered_detections = sorted(detections_by_value_and_query_id,reverse=True)
    
    for key in ordered_detections:
        query_id, image_detected, x1, y1, width, height, value, query_class = detections_by_value_and_query_id[key]
        if float(value)>=params.th_value:
            all_detections_ordered.write('{0} {1} {2} {3} {4} {5} {6} {7}'.format(query_id, image_detected, x1, y1, width, height, value, query_class))
