import os

import argparse

from calculate_AP_class import *

import json

import pandas as pd

from query_finder_class import *


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    #parser.add_argument('-dataset_name', help='dataset name', type=str, choices=['DocExplore', 'flickrlogos_47'], default='flickrlogos_47')
    #parser.add_argument('-coco_images', help='image directory in coco format', type=str, default = '/mnt/BE6CA2E26CA294A5/Datasets/flickrlogos_47_COCO/images/train')
    #parser.add_argument('-annotation_json', help='image directory in coco format', type=str, default = '/mnt/BE6CA2E26CA294A5/Datasets/flickrlogos_47_COCO/annotations/instances_train.json')
    #parser.add_argument('-query_path', help='path to queries', type=str, default = '/mnt/BE6CA2E26CA294A5/Datasets/flickrlogos_47_COCO/images/queries_train/')
    parser.add_argument('-model', help='model used for the convolutional features', type=str, choices=['resnet', 'VGG16'], default='VGG16') 
    parser.add_argument('-layer', help='resnet layer(s) used for extraction, they can be:\n for VGG: {0}\n for resnet:{1}\n For multiple layers, a semicolon "," can be used to separate '.format(
    'conv1_relu, conv2_block3_out, conv3_block4_out, conv4_block6_out, conv5_block3_out',
    'block2_conv2,block3_conv3, block4_conv3, block5_conv3'), type=str, default='block3_conv3')     #parser.add_argument('-feat_savedir', help='directory of features database', type=str, default='/home/jeancherubini/Documents/feature_maps')
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

    #Create dirs for ps task
    if not os.path.isdir('{0}/{1}/{2}/{3}/AP'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer, params.principal_components)):
        os.makedirs('{0}/{1}/{2}/{3}/AP'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer, params.principal_components))

    #All instances AP per class for ps task
    file_all_ap = open('{0}/{1}/{2}/{3}/AP/all_AP.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer, params.principal_components), 'w')
    file_all_ap.write('class\tinstance\theight\twidth\tAP\n')
    file_all_ap.close()

    #Mean AP per class for ps task
    file_all_ap_class = open('{0}/{1}/{2}/{3}/AP/mean_AP_class.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer, params.principal_components), 'w')
    file_all_ap_class.write('class\tmean_height\tmean_width\tmAP\n')

    #Create dirs for ir task
    if not os.path.isdir('{0}/{1}/{2}/{3}/AP_ir'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer, params.principal_components)):
        os.makedirs('{0}/{1}/{2}/{3}/AP_ir'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer, params.principal_components))

    file_all_ap_ir = open('{0}/{1}/{2}/{3}/AP_ir/all_AP.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer, params.principal_components), 'w')
    file_all_ap_ir.write('class\tinstance\theight\twidth\tAP\n')
    file_all_ap_ir.close()

    #Mean AP per class for ps task
    file_all_ap_class_ir = open('{0}/{1}/{2}/{3}/AP_ir/mean_AP_class_ir.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer, params.principal_components), 'w')
    file_all_ap_class_ir.write('class\tmin\tmax\tmAP\n')



    query_classes = os.listdir(params.query_path)




    AP_calculator = AP_calculator_class()
    

    for query_class in query_classes:
        instances = os.listdir('{0}/{1}'.format(params.query_path, query_class))
        for query_instance in instances:
            try:
                finder = query_finder()
                query = finder.get_query(params, query_class, query_instance)
                _, query_height, query_width, _ = query.shape
                #get detections file ordered 
                AP_calculator.get_ordered_detections(params, query_class, query_instance)

                #plot and calculate detection benchmarcks
                #AP_calculator.plt_top_detections(params, query_class, query_instance)
                AP_calculator.calculate_query_ps(params, query_class, query_instance)
                file_ap = open('{0}/{1}/{2}/{3}/AP/{4}/{5}.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer, params.principal_components, query_class, query_instance.replace('.png', '').replace('.jpg','')), 'r') 
                file_all_ap = open('{0}/{1}/{2}/{3}/AP/all_AP.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer, params.principal_components), 'a')
                file_all_ap.write('{0}\t{1}\t{2}\t{3}\t{4}\n'.format(query_class, query_instance, query_height, query_width, file_ap.readline()))
                file_all_ap.close()
                file_ap.close()

                #calculate imaghe benchmarks 
                #AP_calculator.plt_top_detections_ir(params, query_class, query_instance)
                AP_calculator.calculate_query_ir(params, query_class, query_instance)
                file_ap_ir = open('{0}/{1}/{2}/{3}/AP_ir/{4}/{5}.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer, params.principal_components, query_class, query_instance.replace('.png', '').replace('.jpg','')), 'r') 
                file_all_ap_ir = open('{0}/{1}/{2}/{3}/AP_ir/all_AP.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer, params.principal_components), 'a')
                file_all_ap_ir.write('{0}\t{1}\t{2}\t{3}\t{4}\n'.format(query_class, query_instance, query_height, query_width, file_ap_ir.readline()))
                file_all_ap_ir.close()
                file_ap_ir.close()
                

                
            except:
                print('Detections file for query class {0} instance {1} not found'.format(query_class, query_instance))
    

    AP_calculator.create_all_dataset_detections_file(params)
    

    #file_all_ap.close()



    #Calculate mean AP and time
    if not os.path.isfile('{0}/{1}/summary_file.txt'.format(params.feat_savedir, params.dataset_name)):
        #Build summary file
        summary_file = open('{0}/{1}/summary_file.txt'.format(params.feat_savedir, params.dataset_name), 'w')
        summary_file.close()

    #Open summary file
    summary_file = open('{0}/{1}/summary_file.txt'.format(params.feat_savedir, params.dataset_name), 'a')

    #mean AP for detection
    file_all_AP_pandas = pd.read_csv('{0}/{1}/{2}/{3}/AP/all_AP.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer, params.principal_components), sep="\t", header='infer').rename(columns={'AP':'{0} ({1})'.format(params.model + '_' + params.layer, params.principal_components)})
    print(file_all_AP_pandas)
    mean = file_all_AP_pandas.mean()['{0} ({1})'.format(params.model + '_' + params.layer, params.principal_components)]

    #mean AP for ir
    file_all_AP_pandas_ir = pd.read_csv('{0}/{1}/{2}/{3}/AP_ir/all_AP.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer, params.principal_components), sep="\t", header='infer').rename(columns={'AP':'{0} ({1})'.format(params.model + '_' + params.layer, params.principal_components)})
    mean_ir = file_all_AP_pandas_ir.mean()['{0} ({1})'.format(params.model + '_' + params.layer, params.principal_components)]

   

    #time mean
    file_all_time_pandas = pd.read_csv('{0}/{1}/{2}/{3}/detections/time.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer, params.principal_components), sep="\t",names = ['Instance' , 'Layer'],  )
    mean_time = file_all_time_pandas.mean()[0]


    #Calculations by class
    #ps

    means_by_class = file_all_AP_pandas.groupby('class').mean()

    #ir
    means_by_class_ir = file_all_AP_pandas_ir.groupby('class').mean()
    mins_by_class_ir = file_all_AP_pandas_ir.groupby('class').min()
    maxes_by_class_ir = file_all_AP_pandas_ir.groupby('class').max()




    for class_ in means_by_class['{0} ({1})'.format(params.model + '_' + params.layer, params.principal_components)].keys():
        file_all_ap_class.write('{0}\t{1:.3f}\t{2:.3f}\t{3:.3f}\n'.format(class_, means_by_class['height'][class_], means_by_class['width'][class_],  means_by_class['{0} ({1})'.format(params.model + '_' + params.layer, params.principal_components)][class_]
        ))
        file_all_ap_class_ir.write('{0}\t{1:.3f}\t{2:.3f}\t{3:.3f}\n'.format(class_,means_by_class_ir['{0} ({1})'.format(params.model + '_' + params.layer, params.principal_components)][class_],
        mins_by_class_ir['{0} ({1})'.format(params.model + '_' + params.layer, params.principal_components)][class_],
        maxes_by_class_ir['{0} ({1})'.format(params.model + '_' + params.layer, params.principal_components)][class_]))

    file_all_ap_class.close()
    file_all_ap_class_ir.close()

    summary_file.write('{0} ({1})\t{2:.4f}\t{3:.1f}\t{4:.4f}\n'.format(params.model + '_' + params.layer, params.principal_components, mean, mean_time, mean_ir))
    summary_file.close()

    if params.dataset_name == 'DocExplore':
        AP_calculator.DocExplore_task_transformation(params)
        os.chdir('{0}/{1}/evaluation_kit_v2'.format(params.feat_savedir, params.dataset_name))
        os.system('./main --h')

        os.system('./main --task ps --iou 0.5 --in_file {0} --out_file {1}'.format('{0}/{1}/{2}/{3}/detections/ps_for_DocExplore.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer, params.principal_components),
        '{0}/{1}/{2}/{3}/detections/ps_for_DocExplore_results.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer, params.principal_components)
        ))

        os.system('./main --task im --iou 0.5 --in_file {0} --out_file {1}'.format(
        '{0}/{1}/{2}/{3}/detections/im_for_DocExplore.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer, params.principal_components),
        '{0}/{1}/{2}/{3}/detections/im_for_DocExplore_results.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer, params.principal_components)
        ))

        file_kit_result = pd.read_csv('{0}/{1}/{2}/{3}/detections/ps_for_DocExplore_results.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer, params.principal_components), skiprows=1, nrows=1, sep=',', names=['mAP', 'min AP', 'max AP']).replace('mAP=','').replace('min_AP=','').replace('max_AP=','')
        print(file_kit_result)

        file_kit_result_ir = pd.read_csv('{0}/{1}/{2}/{3}/detections/im_for_DocExplore_results.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer, params.principal_components), skiprows=1, nrows=1, sep=',', names=['mAP', 'min AP', 'max AP']).replace('mAP=','').replace('min_AP=','').replace('max_AP=','')
        print(file_kit_result_ir)

    