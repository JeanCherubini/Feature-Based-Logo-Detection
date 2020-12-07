import os

import argparse

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset_name', help='dataset name', type=str, choices=['DocExplore', 'flickrlogos_47'], default='flickrlogos_47')
    parser.add_argument('-coco_images', help='image directory in coco format', type=str, default = '/mnt/BE6CA2E26CA294A5/Datasets/flickrlogos_47_COCO/images/train')
    parser.add_argument('-annotation_json', help='image directory in coco format', type=str, default = '/mnt/BE6CA2E26CA294A5/Datasets/flickrlogos_47_COCO/annotations/instances_train.json')
    parser.add_argument('-query_path', help='path to queries', type=str, default = '/mnt/BE6CA2E26CA294A5/Datasets/flickrlogos_47_COCO/images/queries_train/')
    parser.add_argument('-model', help='model used for the convolutional features', type=str, choices=['resnet', 'VGG16'], default='VGG16') 
    parser.add_argument('-layer', help='resnet layer used for extraction', type=str, choices=['conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out', 'block3_conv3', 'block4_conv3', 'block5_conv3'], default='block3_conv3') 
    parser.add_argument('-feat_savedir', help='directory of features database', type=str, default='/home/jeancherubini/Documents/feature_maps')
    parser.add_argument('-th_value', help='threshhold value to keep image', type=float, default=0.5)

    params = parser.parse_args()    

    file_all_ap = open('{0}/{1}/AP/all_AP.txt'.format(params.feat_savedir, params.dataset_name), 'w')

    query_classes = os.listdir(params.query_path)
    for query_class in query_classes:
        instances = os.listdir('{0}/{1}'.format(params.query_path, query_class))
        for query_instance in instances:
            command_queries = 'python calculate_AP_query.py -dataset_name {0} -coco_images {1} -annotation_json {2} -query_path {3} -query_class {4} -query_instance {5} -th_value {6}'.format(params.dataset_name, params.coco_images, params.annotation_json, params.query_path, query_class, query_instance, params.th_value) 
            os.system(command_queries)
            file_ap = open('{0}/{1}/AP/{2}/{3}.txt'.format(params.feat_savedir, params.dataset_name, query_class, query_instance.replace('.png', '')), 'r')
            AP = file_ap.readline()
            file_ap.close()
            file_all_ap.write(AP+'\n')
    
    file_all_ap.close()