import os

import argparse

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset_name', help='dataset name', type=str, choices=['DocExplore', 'flickrlogos_47'], default='flickrlogos_47')
    parser.add_argument('-model', help='model used for the convolutional features', type=str, choices=['resnet', 'VGG16'], default='VGG16') 
    parser.add_argument('-layer', help='resnet layer used for extraction', type=str, choices=['conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out', 'block3_conv3', 'block4_conv3', 'block5_conv3'], default='block3_conv3') 
    parser.add_argument('-feat_savedir', help='directory of features database', type=str, default='/home/jeancherubini/Documents/feature_maps')

    params = parser.parse_args()    

    

    #Open all detections document
    all_detections_ordered = open('{0}/{1}/{2}/detections/all_detections_ordered.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer),'r')

    #Create dict to group by query for pattern spotting
    detections_by_query_id = {}

    for row in all_detections_ordered:
        #image retrieval
        query_id, image_detected, x1, y1, width, height, value, query_class = row.split(' ') 
        x2 = int(x1)+int(width)
        y2 = int(y1)+int(height)
        try:
            detections_by_query_id[query_id]+= ('{0}-{1}-{2}-{3}-{4} '.format(image_detected, x1, y1, x2, y2))
        except:
            detections_by_query_id[query_id] = '{0}-{1}-{2}-{3}-{4} '.format(image_detected, x1, y1, x2, y2)

    
    #open file to sav ps
    ps_for_DocExplore = open('{0}/{1}/{2}/detections/ps_for_DocExplore.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer),'w')

    for query in detections_by_query_id.keys():
        ps_for_DocExplore.write('{0}:{1}\n'.format(query, detections_by_query_id[query]))

    ps_for_DocExplore.close()