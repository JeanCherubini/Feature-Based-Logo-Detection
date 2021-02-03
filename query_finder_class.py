
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

from skimage import measure
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
    n, height, width, channels = heatmaps.shape
    height_query, width_query , _ = query.shape

    x_elimination_border_query = int(width_query/2)
    y_elimination_border_query = int(height_query/2)

    canvas = np.zeros_like(heatmaps)
    
    #delete original padding
    for hmap_index in range(n):
        o_height, o_width, _ = original_image_sizes[hmap_index]
        extracted_image = heatmaps[hmap_index, y_elimination_border_query:o_height-y_elimination_border_query, x_elimination_border_query:o_width - x_elimination_border_query, :]
        
        canvas[hmap_index, y_elimination_border_query:o_height-y_elimination_border_query, x_elimination_border_query:o_width - x_elimination_border_query, :] = extracted_image

    heatmaps = tf.convert_to_tensor(canvas)
    #print('Time on deleting borders: {:.3f}'.format(time()-t_deletion))
    return heatmaps

def get_p_maximum_values(image_ids, heatmaps, query, p, is_split):
    n, height, width, channels = heatmaps.shape
    height_query, width_query, _= query.shape
    x_deletion_query = int(width_query/2)
    y_deletion_query = int(height_query/2)
    
    #print(np.unravel_index(np.argmax(heatmaps), heatmaps.shape))
    p_points = []

    for hmap_index in range(n):
        current_hmap = np.array(heatmaps[hmap_index])


        for p_num in range(p):
            #Get maximum values
            y_max, x_max, _ = np.unravel_index(np.argmax(current_hmap), current_hmap.shape)

            maximum_value = np.max(current_hmap)

            #Get coordinates for box deletion
            x_del_begin = x_max - x_deletion_query
            
            y_del_begin = y_max - y_deletion_query

            '''
            #Show points rescued
            fig, axs = plt.subplots(1, 1, sharey=False, figsize=(25,15))
            axs.imshow(current_hmap)
            cntr = Rectangle((x_max, y_max), 3, 3, edgecolor='b', facecolor="r")
            axs.add_patch(cntr)
            rect = Rectangle((x_del_begin, y_del_begin), width_query, height_query, edgecolor='g', facecolor="none")
            axs.add_patch(rect)


            left_point = Rectangle((x_del_begin, y_del_begin), 2, 2, edgecolor='g', facecolor="r")
            axs.add_patch(left_point)

            right_point = Rectangle((x_del_begin + width_query, y_del_begin + height_query), 2, 2, edgecolor='g', facecolor="r")
            axs.add_patch(right_point)
            print('x_max',x_max, 'y_max',y_max,'bbox',x_del_begin, y_del_begin, height_query, width_query, 'value', maximum_value)
            plt.show()
            '''
            #deletion of box
            current_hmap[y_del_begin:y_del_begin + height_query, x_del_begin:x_del_begin + width_query] = 0
            if not is_split or is_split == 1:
                point = {'image_id':image_ids[hmap_index] ,'x_max':x_max, 'y_max':y_max, 'bbox':[x_del_begin, y_del_begin, height_query, width_query], 'value':maximum_value} 
            
            #Add width of image to detection only if its split number 2
            if is_split == 2:
                point = {'image_id':image_ids[hmap_index] ,'x_max':x_max, 'y_max':y_max, 'bbox':[x_del_begin+width, y_del_begin, height_query, width_query], 'value':maximum_value} 

            p_points.append(point)
    
    return np.array(p_points)

    
def get_top_images(p_points, global_top_percentage, in_image_top_porcentage):
    #Sort points by value
    sorted_p_points = sorted(p_points, key = lambda i: i['value'], reverse=True)
    
    #Calculate maximum value obtained
    #max_value = sorted_p_points[0]['value']
    #limit_value = max_value-max_value*(global_top_percentage)/100
    
    #Get top points keeping the global_top_percentage calculated from the maximum found in all images
    top_points = [p_point for p_point in sorted_p_points if p_point['value']>0]

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

    height_query, width_query,_= query.shape
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


class query_finder():


    
    def search_query(self, params, query_class, query_instance):
        #check if result already exists

            if(os.path.isfile('{0}/{1}/{2}/{3}/detections/{4}/{5}.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer, params.principal_components, query_class, query_instance.replace('.png','').replace('.jpg','')))):
                print('Results for {} already exist!'.format(query_instance.replace('.png','').replace('.jpg','')))
                return 0

            #if False:
            #    print()

            elif not os.path.isfile('{0}/{1}/{2}/{3}/detections/time.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer, params.principal_components)):
                #create folder for results
                if not os.path.isdir(params.feat_savedir + '/' + params.dataset_name):
                    os.mkdir(params.feat_savedir +'/' + params.dataset_name)

                if not os.path.isdir(params.feat_savedir + '/' + params.dataset_name + '/' + params.model + '_' + params.layer +'/' + str(params.principal_components) + '/detections'):
                    os.mkdir(params.feat_savedir +'/' + params.dataset_name + '/' + params.model + '_' + params.layer +'/' + str(params.principal_components) + '/detections')
            
                #Create file for times 
                time_file = open('{0}/{1}/{2}/{3}/detections/time.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer, params.principal_components),'w')
                time_file.close()
        
            else: 
                #Open time file
                time_file = open('{0}/{1}/{2}/{3}/detections/time.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer, params.principal_components),'a')


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

                if(params.principal_components>=1):
                    #PCA model
                    pca_dir = params.feat_savedir + '/' + params.dataset_name + '/' + params.model + '_' + params.layer + '/' + str(params.principal_components) +'/PCA/'
                    pca = pk.load(open(pca_dir + "/pca_{}.pkl".format(params.principal_components),'rb'))

                #Queryu procesing
                features_query = intermediate_model(query, training=False)

                b, height, width, channels = features_query.shape

                #features reshaped for PCA transformation
                features_reshaped_PCA = tf.reshape(features_query, (b*height*width,channels))
                
                if(params.principal_components>=1):                
                    #PCA
                    pca_features = pca.transform(features_reshaped_PCA)
                else:
                    pca_features = features_reshaped_PCA

                #l2_normalization        
                pca_features = tf.math.l2_normalize(pca_features, axis=-1, 
                                epsilon=1e-12, name=None)

                #Go back to original shape
                final_query_features = tf.reshape(pca_features, (height, width,pca_features.shape[-1]))


                #Resize big queries
                height_feat_query, width_feat_query, channels_feat_query = final_query_features.shape


                while width_feat_query>50 or height_feat_query>50:
                    final_query_features = tf.image.resize(final_query_features, [int(height_feat_query*0.75), int(width_feat_query*0.75)], preserve_aspect_ratio = True)
                    height_feat_query, width_feat_query, channels_feat_query = final_query_features.shape
                    print('query_shape resized:', height_feat_query, width_feat_query, channels_feat_query)
         

                #Casting ino float32 dtype
                final_query_features = tf.dtypes.cast(final_query_features, tf.float32)

                #Expand dims for convolutions
                final_query_features = tf.expand_dims(final_query_features, axis=3)


                
                #image_features directory
                image_feat_savedir = params.feat_savedir + '/'+ params.dataset_name + '/' + params.model + '_' + params.layer + '/' + str(params.principal_components)

                #cant of batches on database
                files_in_features_dir = os.listdir(image_feat_savedir)
                cant_of_batches = 0
                for file_ in files_in_features_dir:
                    if('features' in file_):
                        cant_of_batches +=1
                #recover shape query
                query = tf.squeeze(query)




                t_inicio = time()


                #Get maximum possible convolution value
                max_possible_value = tf.nn.convolution(tf.expand_dims(tf.squeeze(final_query_features),axis=0), final_query_features, padding = 'VALID', strides=[1,1,1,1])
                #Search query in batches of images
                for batch_counter in range(cant_of_batches):
                    try:
                        print('Processing Batch: {0} for query {1}'.format(batch_counter, query_instance))
                        t_batch = time()

                        #load batch of features and the ids of the images 
                        data = np.load(image_feat_savedir + '/features_{}.npy'.format(batch_counter), allow_pickle=True)
                        #print('Time in loading data {}'.format(time()-t_batch))
                        image_ids = data.item().get('image_ids')
                        features = data.item().get('features')
                        annotations = data.item().get('annotations')
                        is_split = data.item().get('is_split')

                        
                        #list of original batch image sizes without padding
                        original_image_sizes = train_images.load_image_batch(image_ids, params.model)['original_sizes']

                        #shape of the batch with padding
                        original_batches, original_height, original_width, original_channels = train_images.load_image_batch(image_ids, params.model)['padded_batch_size']

                        t_conv = time()

                        #Convolution of features of the batch and the query
                        features = tf.convert_to_tensor(features)
                        features = tf.dtypes.cast(features, tf.float32)

                        #print('features shape:{0} \nquery_shape: {1}'.format(features.shape, final_query_features.shape))

                        #convolution between feature batch of images and features of the query 
                        heatmaps = tf.nn.convolution(features, final_query_features, padding = 'SAME', strides=[1,1,1,1])

                        

                        #Normalization by max possible value
                        heatmaps = heatmaps/max_possible_value
                        #print('time on convolutions: {:.3f}'.format(time()-t_conv))

    
                        #interpolation to original image shapes, halving the sizew if it is a split
                        if not(is_split):
                            heatmaps = tf.image.resize(heatmaps, (original_height, original_width), method=tf.image.ResizeMethod.BICUBIC)
                        if is_split:
                            heatmaps = tf.image.resize(heatmaps, (original_height, int(original_width/2)), method=tf.image.ResizeMethod.BICUBIC)


                    
                        #Deletion of heatmap borders, for treating border abnormalities due to padding in the images
                        heatmaps = delete_border_values(heatmaps, original_image_sizes, query)

                        '''
                        #Visualize heatmap
                        for i in range(heatmaps.shape[0]):
                            annotations_image = annotations[i]
                            annotation_labels = annotations_image[:,-1]
                            if query_class_num in annotation_labels:
                                contours = measure.find_contours(np.asarray(tf.squeeze(heatmaps[i])), 0.8)
                                # Display the image and plot all contours found
                                fig, (ax,ax2) = plt.subplots(2,1)
                                ax.imshow(np.asarray(tf.squeeze(heatmaps[i])),cmap='Greys_r')
                                img_load = train_images.load_image_batch(image_ids, params.model)['padded_images']
                                img_correct = img_load[i]/255
                                ax2.imshow(img_correct)
                                print(i)

                                for contour in contours:
                                    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

                                ax.axis('image')
                                ax.set_xticks([])
                                ax.set_yticks([])
                                plt.show()
                                
                                
                                t_points=time()
                                #create db with all the maximum points found 
                                p_points = get_p_maximum_values(image_ids, heatmaps, query, params.p)
                        '''  
                                
                    
                        
                        t_points=time()
                        #create db with all the maximum points found 
                        if(batch_counter == 0):
                            p_points = get_p_maximum_values(image_ids, heatmaps, query, params.p, is_split)
                        else:
                            p_points = np.concatenate( (p_points, get_p_maximum_values(image_ids, heatmaps, query, params.p, is_split)) )
                        #print('Time searching points: {}'.format(time()-t_points))
                                                

                        print('Batch {0} processed in {1}'.format(batch_counter, time()-t_batch))
                    except:
                        print('Batch {} missing'.format(batch_counter))
                    
                    
                    #if batch_counter==3:
                    #    break

                t_procesamiento = time()-t_inicio
                time_file.write('{0}\t{1}\n'.format(query_instance, t_procesamiento))
                time_file.close()
                print('t_procesamiento', t_procesamiento)

                try:
                    #Get top porcentaje of sorted id images and their detections         
                    top_images_ids, top_images_detections = get_top_images(p_points,100,100)

                    
                    #create folder for results
                    if not os.path.isdir(params.feat_savedir + '/' + params.dataset_name):
                        os.mkdir(params.feat_savedir +'/' + params.dataset_name)

                    if not os.path.isdir(params.feat_savedir + '/' + params.dataset_name + '/' + params.model + '_' + params.layer +'/' + str(params.principal_components) + '/detections'):
                        os.mkdir(params.feat_savedir +'/' + params.dataset_name + '/' + params.model + '_' + params.layer +'/' + str(params.principal_components) + '/detections')

                    if not os.path.isdir(params.feat_savedir + '/' + params.dataset_name + '/' + params.model + '_' + params.layer +'/' + str(params.principal_components) + '/detections/'+query_class):
                        os.mkdir(params.feat_savedir + '/' + params.dataset_name + '/' + params.model + '_' + params.layer +'/' + str(params.principal_components) + '/detections/'+query_class)

                    results = open('{0}/{1}/{2}/{3}/detections/{4}/{5}.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer, params.principal_components,  query_class, query_instance.replace('.png','').replace('.jpg','')),'w')
                    #create figure to show query
                        
                    
                    
                    for id_ in top_images_ids:    
                        
                        #get detections for this image
                        bounding_box, values = get_bounding_boxes([id_], top_images_detections, query)
                    
                        for j in range(len(bounding_box[id_])):
                            x1, y1, height, width = bounding_box[id_][j]
                            value = values[id_][j]
                            if not ([x1, y1, height, width]==[0 ,0 , 0 ,0]):
                                results_text = '{0} {1} {2} {3} {4} {5:.3f} {6}\n'.format(id_, x1, y1, height, width, value,  query_class_num)
                                results.write(results_text)
                        '''    
                        for bbox in bounding_box[id_]:
                            x1, y1, height, width = bbox
                            if not ([x1, y1, height, width]==[0 ,0 , 0 ,0]):
                                results_text = '{0} {1} {2} {3} {4} {5}\n'.format(id_, x1, y1, height, width, query_class_num)
                                results.write(results_text)
                        '''
                    results.close()
                    return 1
                except:
                    if not(os.path.isfile('{0}/{1}/{2}/{3}/detections/error_detection.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer, params.principal_components))):
                        errors = open('{0}/{1}/{2}/{3}/detections/error_detection.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer, params.principal_components),'w')
                        errors.close()
                    errors = open('{0}/{1}/{2}/{3}/detections/error_detection.txt'.format(params.feat_savedir, params.dataset_name, params.model + '_' + params.layer, params.principal_components),'a')
                    errors.write('Error finding detections for query class {} instance {}\n'.format(query_class, query_instance.replace('.png','').replace('.jpg','')))
                    print("No se encontraron puntos suficientes")
                    return 0


