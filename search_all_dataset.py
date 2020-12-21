
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

def make_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_layer_model(model,layer_name):
    #Resnet interesting layers
    #conv1_relu, conv2_block3_out, conv3_block4_out, conv4_block6_out, conv5_block3_out
    return tf.keras.Model(model.inputs, model.get_layer(layer_name).output)

def get_query(query_path, query_class, instance):
    query = imread(query_path + '/' + query_class + '/' + instance )/255
    query = query[:,:,:3]
    return query

def delete_border_values(heatmaps, original_image_sizes, query):
    t_deletion = time()
    n, width, height, channels = heatmaps.shape
    width_query, height_query, _= query.shape

    x_elimination_border_query = int(width_query/2)
    y_elimination_border_query = int(height_query/2)

    canvas = np.zeros_like(heatmaps)
    
    #delete original padding
    for hmap_index in range(n):
        o_width, o_height, _ = original_image_sizes[hmap_index]
        extracted_image = heatmaps[hmap_index, x_elimination_border_query:o_width - x_elimination_border_query, y_elimination_border_query:o_height-y_elimination_border_query, :]
        
        canvas[hmap_index, x_elimination_border_query:o_width - x_elimination_border_query, y_elimination_border_query:o_height-y_elimination_border_query, :] = extracted_image
    heatmaps = tf.convert_to_tensor(canvas)
    print('Time on deleting borders: {:.3f}'.format(time()-t_deletion))
    return heatmaps


def get_p_maximum_values_optimized(image_ids, heatmaps, query, p):
    t1 = time()
    n, width, height, channels = heatmaps.shape
    width_query, height_query, _= query.shape

    #Range to delete from the borders, depends on query shape
    x_deletion_query = math.floor(width_query/2)
    y_deletion_query = math.floor(height_query/2)


    #Copy of heatmaps that is actually modifiable
    heatmaps_modifiable = np.array(heatmaps)

    p_points = []

    for p_num in range(p):
        #Get maximum value for each image in the batch
        reduced =  tf.math.reduce_max(heatmaps_modifiable, axis = (1,2), keepdims=True)

        #if there is a zero-max
        reduced = tf.where(tf.equal(reduced, 0), -1000*tf.ones_like(reduced), reduced)

        #Get mask to find where are the maximum values
        mask = tf.equal(reduced, heatmaps_modifiable)
        
        #Get indexes for the p_points
        indexes = tf.where(mask)
       
        max_values = tf.squeeze(reduced)
        max_locations = tf.squeeze(indexes)

        for dim in range(len(max_locations)):
            try:
                if(max_values[dim].numpy()>0):
                    y_begin = (max_locations[dim][1]-y_deletion_query).numpy()
                    if y_begin<0:
                        y_begin=0

                    y_end = (max_locations[dim][1]+y_deletion_query).numpy()
                    if y_end>=width:
                        y_end=width

                    x_begin = (max_locations[dim][2]-x_deletion_query).numpy()
                    if x_begin<0:
                        x_begin=0
                    
                    x_end = (max_locations[dim][2]+x_deletion_query).numpy()
                    if x_end>=height:
                        x_end=height

                    heatmaps_modifiable[dim, y_begin:y_end, x_begin:x_end, 0] = 0

                    point = {'image_id':image_ids[dim] ,'x_max':max_locations[dim][1].numpy(), 'y_max':max_locations[dim][2].numpy(), 'bbox':[x_begin, y_begin, width_query, height_query],  'value':max_values[dim].numpy()} 
                    p_points.append(point)    
            except:
                point = {'image_id':-1 ,'x_max':-1, 'y_max':-1, 'value':-1} 
                p_points.append(point)    
                print('No se encontro punto maximo')
                continue
    print('time in finding points: {:.3f}'.format(time()-t1))
    return np.array(p_points)
    
def get_top_images(p_points, global_top_percentage, in_image_top_porcentage):
    #Sort points by value
    sorted_p_points = sorted(p_points, key = lambda i: i['value'], reverse=True)
    
    #Calculate maximum value obtained
    max_value = sorted_p_points[0]['value']
    limit_value = max_value-max_value*(global_top_percentage)/100
    
    #Get top points keeping the global_top_percentage calculated from the maximum found in all images
    top_points = [p_point for p_point in sorted_p_points if p_point['value']>=limit_value]

    #Group all points by the imaghe they belong to
    grouped_by_image = defaultdict(list)

    for item in top_points:
        grouped_by_image[item['image_id']].append({'x_max':item['x_max'],'y_max':item['y_max'],'bbox':item['bbox'],'value':item['value']})

    #Filter the top detections in each image, keeping the in_image_top_porcentage 
    grouped_by_image_filtered_top = {}

    for key in grouped_by_image.keys():
        values_this_image_id = [det['value'] for det in grouped_by_image[key]]
        max_this_image_id = max(values_this_image_id)
        detections_to_save = [det for det in grouped_by_image[key] if det['value']>=max_this_image_id-max_this_image_id*(in_image_top_porcentage)/100]
        grouped_by_image_filtered_top[key]=detections_to_save


    return grouped_by_image_filtered_top.keys(), grouped_by_image_filtered_top


def get_bounding_boxes(top_images_ids, top_images_detections, query):

    width_query, height_query, _= query.shape
    bboxes = {}
    values = {}
    
    for id_ in top_images_ids:
        bboxes_this_image = [] 
        values_this_image = [] 
        for detection in top_images_detections[id_]:
            bbox = detection['bbox']
            bboxes_this_image.append(bbox)
            value = detection['value']
            values_this_image.append(value)

        bboxes[id_] = np.array(bboxes_this_image)
        values[id_] = np.array(values_this_image)

    return bboxes, values

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset_name', help='dataset name', type=str, choices=['DocExplore', 'flickrlogos_47'], default='flickrlogos_47')
    parser.add_argument('-coco_images', help='image directory in coco format', type=str, default = '/mnt/BE6CA2E26CA294A5/Datasets/flickrlogos_47_COCO/images/train')
    parser.add_argument('-annotation_json', help='image directory in coco format', type=str, default = '/mnt/BE6CA2E26CA294A5/Datasets/flickrlogos_47_COCO/annotations/instances_train.json')
    parser.add_argument('-query_path', help='path to queries', type=str, default = '/mnt/BE6CA2E26CA294A5/Datasets/flickrlogos_47_COCO/images/queries_train/')
    parser.add_argument('-feat_savedir', help='directory of features database', type=str, default='/home/jeancherubini/Documents/feature_maps')
    parser.add_argument('-principal_components', help='amount of components kept (depth of feature vectors)', type=int, default=64)
    parser.add_argument('-model', help='model used for the convolutional features', type=str, choices=['resnet', 'VGG16'], default='VGG16') 
    parser.add_argument('-layer', help='resnet layer used for extraction', type=str, choices=['conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out', 'block3_conv3', 'block4_conv3', 'block5_conv3'], default='block3_conv3') 
    parser.add_argument('-p', help='max points collected from each heatmap', type=int, default=15) 
    
    params = parser.parse_args()    

    # GPU OPTIONS
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    for query_class in os.listdir(params.query_path):
        for query_instance in os.listdir(params.query_path + '/' + query_class):
            #check if result already exists
            if(os.path.isfile('{0}/{1}/{2}/detections/{3}/{4}.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer, query_class, query_instance.replace('.png','').replace('.jpg','')))):
                print('Results for {} already exist!'.format(query_instance.replace('.png','').replace('.jpg','')))
            else: 
                

                #creation of dataset like coco
                train_images = CocoLikeDataset()
                train_images.load_data(params.annotation_json, params.coco_images)
                train_images.prepare()

                classes_dictionary = train_images.class_info
                query_class_num = [cat['id'] for cat in classes_dictionary if cat['name']==query_class][0]


                #load desired query
                if query_instance=='random':
                    instances = os.listdir(params.query_path+ '/' + query_class)
                    num_instances = len(os.listdir(params.query_path+ '/' + query_class))
                    instance = instances[np.random.randint(0,num_instances)]
                    query = get_query(params.query_path, query_class, instance)
                else:
                    query = imread(params.query_path + '/' + query_class + '/' + query_instance)[:,:,:3]/255


                #Expand dims to batch
                query = tf.expand_dims(query, axis=0)

                


                #base model
                if(params.model == 'resnet'):
                    model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', 
                                                        input_tensor=None, input_shape=None, pooling=None, classes=1000)
                elif(params.model == 'VGG16'):
                    model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=None,
                                                        pooling=None, classes=1000, classifier_activation='softmax')
                else:
                    raise Exception('Please use a valid model')

                #intermediate_model
                intermediate_model = get_layer_model(model, params.layer)

                #PCA model
                pca_dir = params.feat_savedir + '/' + params.dataset_name + '/' + params.model + '_' + params.layer +'/PCA/'
                pca = pk.load(open(pca_dir + "/pca_{}.pkl".format(params.principal_components),'rb'))

                #Queryu procesing
                features_query = intermediate_model(query, training=False)

                b, width, height, channels = features_query.shape
                
                #features reshaped for PCA transformation
                features_reshaped_PCA = tf.reshape(features_query, (b*width*height,channels))
                
                #PCA
                pca_features = pca.transform(features_reshaped_PCA)

                #l2_normalization        
                pca_features = tf.math.l2_normalize(pca_features, axis=-1, 
                                epsilon=1e-12, name=None)

                #Go back to original shape
                final_query_features = tf.reshape(pca_features, (width,height,pca_features.shape[-1]))


                #Resize big queries
                width_feat_query, height_feat_query, channels_feat_query = final_query_features.shape


                while width_feat_query>150 or height_feat_query>150:
                    final_query_features = tf.image.resize(final_query_features, [int(width_feat_query*0.75), int(height_feat_query*0.75)], preserve_aspect_ratio = True)
                    width_feat_query, height_feat_query, channels_feat_query = final_query_features.shape
                    print('query_shape resized:', width_feat_query, height_feat_query, channels_feat_query)
         

            
                final_query_features = tf.dtypes.cast(final_query_features, tf.float16)

                #Expand dims for convolutions
                final_query_features = tf.expand_dims(final_query_features, axis=3)


                
                #image_features directory
                image_feat_savedir = params.feat_savedir + '/'+ params.dataset_name + '/' + params.model + '_' + params.layer

                #cant of batches on database
                files_in_features_dir = os.listdir(image_feat_savedir)
                cant_of_batches = 0
                for file_ in files_in_features_dir:
                    if('features' in file_):
                        cant_of_batches +=1
                #recover shape query
                query = tf.squeeze(query)




                t_inicio = time()
                max_possible_value = tf.nn.convolution(tf.expand_dims(tf.squeeze(final_query_features),axis=0), final_query_features, padding = 'VALID', strides=[1,1,1,1])
                #Search query in batches of images
                for batch_counter in range(cant_of_batches):
                    try:
                        print('Processing Batch: {0} for query {1}'.format(batch_counter, query_instance))
                        t_batch = time()

                        data = np.load(image_feat_savedir + '/features_{}.npy'.format(batch_counter), allow_pickle=True)
                        print('Time in loading data {}'.format(time()-t_batch))
                        image_ids = data.item().get('image_ids')
                        features = data.item().get('features')
                        
                        #original image
                        original_image_sizes = train_images.load_image_batch(image_ids)['original_sizes']

                        #original image size
                        original_batches, original_width, original_height, original_channels = train_images.load_image_batch(image_ids)['padded_batch_size']


                        t_conv = time()
                        #Convolution of features of the batch and the query
                        features = tf.convert_to_tensor(features)
                        features = tf.dtypes.cast(features, tf.float16)

                        print('features shape:{0} \n query_shape: {1}'.format(features.shape, final_query_features.shape))

                        heatmaps = tf.nn.convolution(features, final_query_features, padding = 'SAME', strides=[1,1,1,1])
                        heatmaps = heatmaps/max_possible_value
                        print('time on convolutions: {:.3f}'.format(time()-t_conv))

                        #interpolation to original image shapes
                        heatmaps = tf.image.resize(heatmaps, (original_width, original_height), method=tf.image.ResizeMethod.BICUBIC)

                        #Deletion of heatmap borders
                        heatmaps = delete_border_values(heatmaps, original_image_sizes, query)


                        if(batch_counter == 0):
                            p_points = get_p_maximum_values_optimized(image_ids, heatmaps, query, params.p)
                        else:
                            p_points = np.concatenate( (p_points, get_p_maximum_values_optimized(image_ids, heatmaps, query, params.p)) )
                        

                        print('Batch {0} processed in {1}'.format(batch_counter, time()-t_batch))
                    except:
                        print('Batch {} missing'.format(batch_counter))
                    
                    
                    #if batch_counter==3:
                    #    break

                t_procesamiento = time()-t_inicio
                print('t_procesamiento', t_procesamiento)


                #Get top porcentaje of sorted id images and their detections         
                top_images_ids, top_images_detections = get_top_images(p_points,100,100)

                

                #Get the detections in bounding box format for each image
                bounding_boxes = get_bounding_boxes(top_images_ids, top_images_detections, query)
                
                #Annotate detections in coco format

                #create folder for results
                if not os.path.isdir(params.feat_savedir + '/' + params.dataset_name):
                    os.mkdir(params.feat_savedir +'/' + params.dataset_name)

                if not os.path.isdir(params.feat_savedir + '/' + params.dataset_name + '/' + params.model + '_' + params.layer + '/detections'):
                    os.mkdir(params.feat_savedir +'/' + params.dataset_name + '/' + params.model + '_' + params.layer + '/detections')

                if not os.path.isdir(params.feat_savedir + '/' + params.dataset_name + '/' + params.model + '_' + params.layer + '/detections/'+query_class):
                    os.mkdir(params.feat_savedir + '/' + params.dataset_name + '/' + params.model + '_' + params.layer + '/detections/'+query_class)

                results = open('{0}/{1}/{2}/detections/{3}/{4}.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer , query_class, query_instance.replace('.png','').replace('.jpg','')),'w')
                #create figure to show query
                    
                
                
                for id_ in top_images_ids:    
                    
                    #get detections for this image
                    bounding_box, values = get_bounding_boxes([id_], top_images_detections, query)
                
                    for j in range(len(bounding_box[id_])):
                        x1, y1, height, width = bounding_box[id_][j]
                        value = values[id_][j]
                        if not ([x1, y1, width, height]==[0 ,0 , 0 ,0]):
                            results_text = '{0} {1} {2} {3} {4} {5:.3f} {6}\n'.format(id_, x1, y1, width, height, value,  query_class_num)
                            results.write(results_text)
                    '''    
                    for bbox in bounding_box[id_]:
                        x1, y1, height, width = bbox
                        if not ([x1, y1, width, height]==[0 ,0 , 0 ,0]):
                            results_text = '{0} {1} {2} {3} {4} {5}\n'.format(id_, x1, y1, width, height, query_class_num)
                            results.write(results_text)
                    '''
                results.close()


            

        

        









if __name__ == '__main__' :
    exit(main())