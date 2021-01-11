import os

import argparse

from utils.COCO_Utils.COCO_like_dataset import CocoLikeDataset 
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset_name', help='dataset name', type=str, choices=['DocExplore', 'flickrlogos_47'], default='flickrlogos_47')
    parser.add_argument('-query_path', help='path to queries', type=str, default = '/mnt/BE6CA2E26CA294A5/Datasets/flickrlogos_47_COCO/images/queries_train/')
    parser.add_argument('-model', help='model used for the convolutional features', type=str, choices=['resnet', 'VGG16'], default='VGG16') 
    parser.add_argument('-layer', help='resnet layer used for extraction', type=str, choices=['conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out', 'block3_conv3', 'block4_conv3', 'block5_conv3'], default='block3_conv3') 
    parser.add_argument('-feat_savedir', help='directory of features database', type=str, default='/home/jeancherubini/Documents/feature_maps')
    parser.add_argument('-principal_components', help='amount of components kept (depth of feature vectors)', type=str, default='64')   
    parser.add_argument('-th_value', help='threshhold value to keep image', type=float, default=0.1)

    params = parser.parse_args()    
    
    all_detections = open('{0}/{1}/{2}/detections/all_detections.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer + '/' + params.principal_components ),'w')
    errors = open('{0}/{1}/{2}/detections/errors.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer + '/' + params.principal_components ) ,'w')

    query_classes = os.listdir(params.query_path)
    for query_class in query_classes:
        instances = os.listdir('{0}/{1}'.format(params.query_path, query_class))
        for query_instance in instances:
            try:
                result_query = open('{0}/{1}/{2}/{3}/detections/{4}/{5}.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer, params.principal_components, query_class, query_instance.replace('.png','').replace('.jpg','')),'r')
                last_row=''
                for row in result_query:
                    if row!=last_row:
                        all_detections.write(query_instance.replace('.png','').replace('.jpg','') + ' ' + row)
                        last_row = row
                result_query.close()
            except:
                errors.write('Error finding detections for query class {} instance {}\n'.format(query_class, query_instance.replace('.png','').replace('.jpg','')))
                continue
    errors.close()
    all_detections.close()

    #Sort detections in a document
    #Open all detections document
    all_detections = open('{0}/{1}/{2}/{3}/detections/all_detections.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer, params.principal_components),'r')
    
    #Open file where detections ordered by value will be written
    all_detections_ordered = open('{0}/{1}/{2}/{3}/detections/all_detections_ordered.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer, params.principal_components),'w')

    detections_by_value_and_query_id = {}

    all_detections_filename = '{0}/{1}/{2}/{3}/detections/all_detections.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer, params.principal_components)
    all_detections_ordered_filename = '{0}/{1}/{2}/{3}/detections/all_detections_ordered.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer, params.principal_components)

    with open(all_detections_filename,'r') as all_detections:
        rows = all_detections.readlines()
        sorted_rows = sorted(rows, key=lambda x: float(x.split()[6]), reverse=True)
        with open(all_detections_ordered_filename,'w') as second_file:
            for row in sorted_rows:
                if(float(row.split()[6])>=params.th_value):
                    all_detections_ordered.write(row)






    '''
    for row in all_detections:
        #image retrieval
        query_id, image_detected, x1, y1, height, width, value, query_class = row.split(' ') 
        detections_by_value_and_query_id[float(value), query_id, x1]=[query_id, image_detected, x1, y1, height, width, value, query_class]
    

    ordered_detections = sorted(detections_by_value_and_query_id,reverse=True)

    for key in ordered_detections.keys():
        value, query_id = key
        print(value, query_id)
        query_id, image_detected, x1, y1, height, width, value, query_class = detections_by_value_and_query_id[key]
        
    '''
    '''
        image = train_images.load_image(int(image_detected))
        fig, ax0 = plt.subplots(1,1)
        ax0.imshow(image)
        rect = Rectangle((int(x1),int(y1)), int(height), int(width), edgecolor='g', facecolor="none")
        ax0.add_patch(rect)
        plt.show()
    

        if float(value)>=params.th_value:
            all_detections_ordered.write('{0} {1} {2} {3} {4} {5} {6} {7}'.format(query_id, image_detected, x1, y1, height, width, value, query_class))
    '''