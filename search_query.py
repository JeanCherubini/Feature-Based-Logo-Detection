
import sys
import tensorflow as tf
from models import resnet#, uv_rois
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

import skimage.io

from PIL import Image, ImageDraw 


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
    dirs = os.listdir(query_path)
    for directory in dirs:
        dir_class = directory.split('_')[0]
        if(dir_class == str(query_class)):
            files = os.listdir(query_path + '/' + directory)
            query = skimage.io.imread(query_path + '/' + directory + '/' + files[instance], )/255
            query = query[:,:,:3]
    return query


def delete_border_values(heatmaps, original_image_sizes, query):
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
    return heatmaps

def get_p_maximum_values(image_ids, heatmaps, query, p):
    n, width, height, channels = heatmaps.shape
    width_query, height_query, _= query.shape

    x_deletion_query = int(width_query/2)
    y_deletion_query = int(height_query/2)

    #print(np.unravel_index(np.argmax(heatmaps), heatmaps.shape))

    p_points = []

    for hmap_index in range(n):
        current_hmap = np.array(heatmaps[hmap_index])    
        for p_num in range(p):
            
            x_max, y_max, _ = np.unravel_index(np.argmax(current_hmap), current_hmap.shape)
            maximum_value = np.max(current_hmap)

            x_del_begin = x_max - x_deletion_query
            y_del_begin = y_max - y_deletion_query

            x_del_end = x_max + x_deletion_query
            y_del_end = y_max + y_deletion_query


            current_hmap[x_del_begin:x_del_end, y_del_begin:y_del_end] = 0

            point = {'image_id':image_ids[hmap_index] ,'x_max':x_max, 'y_max':y_max, 'value':maximum_value} 
            
            p_points.append(point)

            
        return np.array(p_points)

def get_p_maximum_values_optimized(image_ids, heatmaps, query, p):
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
                point = {'image_id':-1 ,'x_max':-1, 'y_max':-1, 'value':0} 
                p_points.append(point)    
                print('No se encontro punto maximo')
                continue

    return np.array(p_points)
    
def get_top_images(p_points, top_percentage):
    #Sort points by value
    sorted_p_points = sorted(p_points, key = lambda i: i['value'], reverse=True)
    print('sorted_p_points', sorted_p_points, '\n')
    
    max_value = sorted_p_points[0]['value']
    limit_value = max_value-max_value*(top_percentage)/100
    
    top_points = [p_point for p_point in sorted_p_points if p_point['value']>=limit_value]
    print('top_points', top_points, '\n')

    grouped_by_image = defaultdict(list)

    for item in top_points:
        grouped_by_image[item['image_id']].append({'x_max':item['x_max'],'y_max':item['y_max'],'bbox':item['bbox'],'value':item['value']})


    print('grouped_by_image', grouped_by_image, '\n' )
    print('keys', grouped_by_image.keys(), '\n')

    return grouped_by_image.keys(), grouped_by_image

def get_bounding_boxes(top_images_ids, top_images_detections, query):

    width_query, height_query, _= query.shape

    query_half_width= int(width_query/2)
    query_half_height = int(height_query/2)
    bboxes = {}
    
    for id_ in top_images_ids:
        bboxes_this_image = [] 
        for detection in top_images_detections[id_]:
            bbox = detection['bbox']
            bboxes_this_image.append(bbox)
        bboxes[id_] = np.array(bboxes_this_image)

    print(bboxes)

    return bboxes




if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('-coco_images', help='image directory in coco format', type=str, default = '/home/jeancherubini/Documents/data/coco_flickrlogos_47/images/train')
    parser.add_argument('-annotation_json', help='image directory in coco format', type=str, default = '/home/jeancherubini/Documents/data/coco_flickrlogos_47/annotations/instances_train.json')
    parser.add_argument('-query_path', help='query_location', type=str, default = '/home/jeancherubini/Documents/data/coco_flickrlogos_47/images/queries_train')
    parser.add_argument('-query_class', help='class of the desired query', type=int, default = 7)
    parser.add_argument('-features_dir', help='directory of features database', type=str, default='/home/jeancherubini/Documents/feature_maps')
    parser.add_argument('-principal_components', help='amount of components kept (depth of feature vectors)', type=int, default=64)
    parser.add_argument('-model', help='model used for the convolutional features', type=str, choices=['resnet', 'VGG16'], default='VGG16') 
    parser.add_argument('-layer', help='resnet layer used for extraction', type=str, choices=['conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out', 'block3_conv3', 'block4_conv3', 'block5_conv3'], default='conv2_block3_out') 
    parser.add_argument('-p', help='max points collected from each heatmap', type=int, default=15) 


    params = parser.parse_args()    

    #creation of dataset like coco
    train_images = CocoLikeDataset()
    train_images.load_data(params.annotation_json, params.coco_images)
    train_images.prepare()


    #load desired query
    instance = 5
    query = get_query(params.query_path, params.query_class, instance)
    print(query.shape)
    

    #Expand dims to batch
    query = tf.expand_dims(query, axis=0)

    # GPU OPTIONS
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


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
    pca_dir = params.features_dir + '/PCA/' + params.model + '_' + params.layer
    pca = pk.load(open(pca_dir + "/pca_{}.pkl".format(params.principal_components),'rb'))

    #Queryu procesing
    features_query = intermediate_model(query, training=False)
    print('features_query', features_query.shape)

    b, width, height, channels = features_query.shape
    
    #features reshaped for PCA transformation
    features_reshaped_PCA = tf.reshape(features_query, (b*width*height,channels))
    print('features reshaped for pca', features_reshaped_PCA.shape)
    
    #PCA
    pca_features = pca.transform(features_reshaped_PCA)
    print('pca features', pca_features.shape)

    #l2_normalization        
    pca_features = tf.math.l2_normalize(pca_features, axis=-1, 
                    epsilon=1e-12, name=None)

    #Go back to original shape
    final_query_features = tf.reshape(pca_features, (width,height,pca_features.shape[-1]))

    final_query_features = tf.expand_dims(final_query_features, axis=3)
    width_feat_query, height_feat_query, channels_feat_query, channels_feat_output_query = query.shape

    print('final_query_features', final_query_features.shape)

    
    #image_features directory
    image_features_dir = params.features_dir + '/' + params.model + '_' + params.layer
    print(image_features_dir)

    #cant of batches on database
    cant_of_batches = len(os.listdir(image_features_dir))

    #recover shape query
    query = tf.squeeze(query)


    #show query
    plt.figure()
    plt.imshow(query)
    plt.show()


    for batch_counter in range(cant_of_batches):     
        print('Processing Batch: {}'.format(batch_counter))


        data = np.load(image_features_dir+'/features_{}.npy'.format(batch_counter), allow_pickle=True)
        image_ids = data.item().get('image_ids')
        features = data.item().get('features')
        
        #original image
        images = train_images.load_image_batch(image_ids)['padded_images']/255
        original_image_sizes = train_images.load_image_batch(image_ids)['original_sizes']
        annotations = train_images.load_annotations_batch(image_ids)

        #original image size
        original_batches, original_width, original_height, original_channels = images.shape


        #Convolution of features of the batch and the query
        features = tf.convert_to_tensor(features)
        heatmaps = tf.nn.convolution(features, final_query_features, padding = 'SAME', strides=[1,1,1,1])
        heatmaps = heatmaps/(width_feat_query*height_feat_query)

        #interpolation to original image shapes
        heatmaps = tf.image.resize(heatmaps, (original_width, original_height), method=tf.image.ResizeMethod.BICUBIC)

        #Deletion of heatmap borders
        heatmaps = delete_border_values(heatmaps, original_image_sizes, query)


        if(batch_counter == 0):
            p_points = get_p_maximum_values_optimized(image_ids, heatmaps, query, params.p)
        else:
            p_points = np.concatenate( (p_points, get_p_maximum_values_optimized(image_ids, heatmaps, query, params.p)) )
        

        print('Batch {} processed'.format(batch_counter))

 
        if batch_counter==5:
            break

    #Get top porcentaje of sorted id images and their detections         
    top_images_ids, top_images_detections = get_top_images(p_points,10)

    #Get the detections in bounding box format for each image
    bounding_boxes = get_bounding_boxes(top_images_ids, top_images_detections, query)

    #Load the top images
    #top_images = train_images.load_image_batch(top_images_ids)['padded_images']/255
    #top_images_annotations = train_images.load_annotations_batch(top_images_ids)

    display = 1
    if display:
        for id_ in top_images_ids:
            #create figure to 
            f, (ax0, ax1) = plt.subplots(1, 2, sharey=False)
            ax0.imshow(query)
            
            #image load
            image = train_images.load_image(id_)
            ax1.imshow(image)
            
            #get ground truth for this image
            annotation = train_images.load_annotations(id_)

            #get detections for this image
            bounding_box = get_bounding_boxes([id_], top_images_detections, query)
            print('bounding_box', bounding_box)
        

            for ann in annotation:
                x1, y1 ,width ,height, label = ann 
                if not ([x1, y1, width, height]==[0 ,0 , 0 ,0]):
                    if(int(params.query_class)==int(label)):         
                        rect = Rectangle((x1,y1), width, height, edgecolor='g', facecolor="none")
                        ax1.add_patch(rect)
                
            for bbox in bounding_box[id_]:
                x1, y1, height, width = bbox
                if not ([x1, y1, width, height]==[0 ,0 , 0 ,0]):
                    rect = Rectangle((x1,y1), width, height, edgecolor='r', facecolor="none")
                    ax1.add_patch(rect)
            
            plt.show()

    top_10 = 1

    if top_10:
        #create figure to 
        fig, ([ax0, ax1, ax2, ax3, ax4], [ax5, ax6, ax7, ax8, ax9, ax10]) = plt.subplots(1, 2, sharey=False)
        axs = ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10 
        plt.imshow(query)
        
        for i in range(10):
            id_ = top_images_ids

            
            
            #image load
            image = train_images.load_image(id_)
            ax1.imshow(image)
            
            #get ground truth for this image
            annotation = train_images.load_annotations(id_)

            #get detections for this image
            bounding_box = get_bounding_boxes([id_], top_images_detections, query)
            print('bounding_box', bounding_box)
        

            for ann in annotation:
                x1, y1 ,width ,height, label = ann 
                if not ([x1, y1, width, height]==[0 ,0 , 0 ,0]):
                    if(int(params.query_class)==int(label)):         
                        rect = Rectangle((x1,y1), width, height, edgecolor='g', facecolor="none")
                        ax1.add_patch(rect)
                
            for bbox in bounding_box[id_]:
                print(bbox)
                x1, y1, height, width = bbox
                if not ([x1, y1, width, height]==[0 ,0 , 0 ,0]):
                    rect = Rectangle((x1,y1), width, height, edgecolor='r', facecolor="none")
                    ax1.add_patch(rect)
            
            plt.show()





