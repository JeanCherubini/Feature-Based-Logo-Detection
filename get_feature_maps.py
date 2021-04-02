
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

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils.COCO_Utils.COCO_like_dataset import CocoLikeDataset 

from models.retinanet import retinanet, load_retinanet

import json

def make_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        

def get_layer_model(model,layer_name):
    #Resnet interesting layers
    #conv1_relu, conv2_block3_out, conv3_block4_out, conv4_block6_out, conv5_block3_out
    return tf.keras.Model(model.inputs, model.get_layer(layer_name).output)

def yield_batch_for_PCA(batches, train_images, intermediate_model):
        batch_counter=0
        for batch in list(batches):
            #original images for training
            images = train_images.load_image_batch(batch, params.model)['padded_images']/255

            print('batch shape', images.shape)

            #features extracted
            features_batch = intermediate_model(images, training=False)

            b, height, width, channels = features_batch.shape
            
            #features reshaped for PCA transformation
            features_reshaped_PCA = tf.reshape(features_batch, (b*height*width,channels))
            print('features reshaped for PCA', features_reshaped_PCA.shape)

            batch_counter+=1
            if batch_counter>params.batches_pca:
                break
                
            yield features_reshaped_PCA
            

        
def split_image(image):
    n, height, width, channels = image.shape

    half_point = int(width/2)

    split_1 = np.zeros([n,height,int(width/2),channels])
    split_2 = np.zeros([n,height,int(width/2),channels])

    split_1 = image[:,:,:half_point,:]
    split_2 = image[:,:,half_point:,:]

    return split_1, split_2

        
class feature_getter_class():
    def get_features(self, params):


        if not os.path.isdir(params.feat_savedir+'/'+params.dataset_name):
            os.makedirs(params.feat_savedir+'/'+params.dataset_name)
        
        
        features_path = params.feat_savedir + '/' + params.dataset_name + '/' + params.model + '_' + params.layer + '/' + str(params.principal_components)

        #Create features path
        if not os.path.isdir(features_path):
            os.makedirs(features_path) 

        pca_path = features_path + '/PCA/' 
        if not os.path.isdir(pca_path):
            os.makedirs(pca_path)

        # GPU OPTIONS
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

        if params.model!='retinanet':
            #base model
            if(params.model == 'resnet'):
                model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', 
                                                        input_tensor=None, input_shape=None, pooling=None, classes=1000)
            if(params.model == 'resnet152'):
                model = tf.keras.applications.ResNet152(include_top=False, weights='imagenet', 
                                                        input_tensor=None, input_shape=None, pooling=None, classes=1000)                                      
            elif(params.model == 'VGG16'):
                model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=None,
                                                    pooling=None, classes=1000, classifier_activation='softmax')
            else:
                raise Exception('Please use a valid model')

            
            #intermediate_model
            intermediate_model = get_layer_model(model, params.layer)


            #creation of dataset like coco
            train_images = CocoLikeDataset()
            train_images.load_data(params.annotation_json, params.coco_images)
            train_images.prepare()

            #image ids sorted by width and total of images
            ids = np.array(train_images.sort_all_ids_by_width())
            total_images = len(ids)
        
            batches = make_chunks(ids, params.batch_size)

            #creacion de PCA
            if(params.principal_components>=1):
                pca = IncrementalPCA(n_components=params.principal_components, copy=False, batch_size = 16)

            
                features_for_pca_training_generator = yield_batch_for_PCA(batches, train_images, intermediate_model)

                for features_for_pca_training in list(features_for_pca_training_generator):
                    print('training PCA model with {} features'.format(features_for_pca_training.shape))
                    pca.partial_fit(features_for_pca_training)



            
                print('variance:', pca.explained_variance_)
                pk.dump(pca, open(pca_path + "/pca_{}.pkl".format(params.principal_components),"wb"))
                #Memory free
                features_for_pca_training = []
                

            batches = make_chunks(ids, params.batch_size)
            batch_counter=0

            #Record errors in the batch_processing
            error_log = open(features_path + '/errors.txt', 'w')


            #Try to transform every batch with initial size
            #set condition for failure iun batch processing
            failed_batches=[]
            failed=False
            for batch in list(batches):
                try:
                    if not failed:
                        print('images in batch', batch)
                        #original image
                        images = train_images.load_image_batch(batch, params.model)['padded_images']/255
                        annotations = train_images.load_annotations_batch(batch)

                        #features extracted
                        features_batch = intermediate_model(images, training=False)

                        b, height, width, channels = features_batch.shape

                        #features reshaped for PCA transformation
                        features_reshaped_PCA = tf.reshape(features_batch, (b*height*width,channels))

                        if(params.principal_components>=1):                   
                            #PCA
                            pca_features = pca.transform(features_reshaped_PCA)
                            #l2_normalization        
                            pca_features = tf.math.l2_normalize(pca_features, axis=-1, 
                                            epsilon=1e-12, name=None)

                            
                            #Go back to original shape
                            features_to_save = tf.reshape(pca_features, (b,height,width,params.principal_components))
                        else:
                            pca_features = features_reshaped_PCA
                            #l2_normalization        
                            pca_features = tf.math.l2_normalize(pca_features, axis=-1, 
                                            epsilon=1e-12, name=None)

                            
                            #Go back to original shape
                            features_to_save = tf.reshape(pca_features, (b,height,width,channels))

                        



                        np.save(features_path + '/features_{}'.format(batch_counter), {'image_ids':batch, 'features':features_to_save, 'annotations':annotations, 'is_split':0})
                        

                        print('batch number:', batch_counter, features_to_save.shape)
                        batch_counter+=1
                    elif(failed):
                        failed_batches = np.concatenate((failed_batches,batch))
                except:
                    failed_batches = batch
                    failed=True
                    continue

            #If fails, try extracting the images that are too big and push them to the end of the cicle.
            big_images = []

            while len(failed_batches)>0:
                batches = make_chunks(failed_batches, params.batch_size)
                failed = False
                for batch in list(batches):
                    try:
                        if not failed:
                            print('images in batch',batch)
                            #original image
                            images = train_images.load_image_batch(batch, params.model)['padded_images']/255
                            annotations = train_images.load_annotations_batch(batch)
                                
                            #features extracted
                            features_batch = intermediate_model(images, training=False)

                            b, height, width, channels = features_batch.shape
                            
                            #features reshaped for PCA transformation
                            features_reshaped_PCA = tf.reshape(features_batch, (b*height*width,channels))

                            if(params.principal_components>=1):                          
                                #PCA
                                pca_features = pca.transform(features_reshaped_PCA)
                                    #l2_normalization        
                                pca_features = tf.math.l2_normalize(pca_features, axis=-1, 
                                                epsilon=1e-12, name=None)

                                #Go back to original shape
                                features_to_save = tf.reshape(pca_features, (b,height, width,params.principal_components))
                            else:
                                pca_features = features_reshaped_PCA
                                #l2_normalization        
                                pca_features = tf.math.l2_normalize(pca_features, axis=-1, 
                                                epsilon=1e-12, name=None)

                                #Go back to original shape
                                features_to_save = tf.reshape(pca_features, (b,height, width, channels))

                            

                            np.save(features_path + '/features_{}'.format(batch_counter), {'image_ids':batch, 'features':features_to_save, 'annotations':annotations, 'is_split':0})

                            print('batch number:', batch_counter, features_to_save.shape)
                            batch_counter+=1

                            #condicion de termino                    
                            if len(failed_batches)<=params.batch_size:
                                failed_batches = []

                        elif failed:                    
                            failed_batches = np.concatenate((failed_batches,batch))
                        


                    except:
                        batch_ids_sorted_height = train_images.sort_ids_by_heigth(batch)
                        print('batch_ids_sorted_height',batch_ids_sorted_height) 
                        big_images.append(batch_ids_sorted_height[-1])
                        print('big_images', big_images)       
                        failed_batches = batch_ids_sorted_height[:-1]
                        print(failed_batches)
                        failed = True
                        continue
                

            #Process each big image separately as single images
            for big_image in big_images:
                try:
                    print('images in batch',big_image)
                    #original image
                    images = train_images.load_image_batch([big_image], params.model)['padded_images']/255
                    annotations = train_images.load_annotations_batch([big_image])

                    split_counter = 1

                    print('images', images.shape)

                    for split in split_image(images):
                        #feparams
                        #It is not working properly because only one split is shown
                        print('split', split.shape)

                        #features extracted
                        features_batch = intermediate_model(split, training=False)

                        b, height, width, channels = features_batch.shape
                        
                        #features reshaped for PCA transformation
                        features_reshaped_PCA = tf.reshape(features_batch, (b*height*width,channels))

                        print('features_reshaped_pca', features_reshaped_PCA.shape)


                        if(params.principal_components>=1):                   
                            #PCA
                            pca_features = pca.transform(features_reshaped_PCA)
                            #l2_normalization        
                            pca_features = tf.math.l2_normalize(pca_features, axis=-1, 
                                            epsilon=1e-12, name=None)

                            #Go back to original shape
                            features_to_save = tf.reshape(pca_features, (b,height,width,params.principal_components))
                            print('features_to_save', features_to_save.shape)

                            
                        else:
                            pca_features = features_reshaped_PCA
                            
                            #l2_normalization        
                            pca_features = tf.math.l2_normalize(pca_features, axis=-1, 
                                            epsilon=1e-12, name=None)

                            #Go back to original shape
                            features_to_save = tf.reshape(pca_features, (b,height,width,channels))

                        np.save(features_path + '/features_{}'.format(batch_counter), {'image_ids':[big_image], 'features':features_to_save, 'annotations':annotations, 'is_split':split_counter})

                        print('batch number:', batch_counter, features_to_save.shape, split_counter)

                        batch_counter += 1
                        split_counter += 1
                except:
                    error_log.write('image with id {} impossible to allocate\n'.format(big_image))
                    print('Fail')
                    continue
        
        return 0
        

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
    #parser.add_argument('-dataset_name', help='dataset name', type=str, choices=['DocExplore', 'flickrlogos_47'], default='flickrlogos_47')
    #parser.add_argument('-coco_images', help='image directory in coco format', type=str, default = '/mnt/BE6CA2E26CA294A5/Datasets/flickrlogos_47_COCO/images/train')
    #parser.add_argument('-annotation_json', help='image directory in coco format', type=str, default = '/mnt/BE6CA2E26CA294A5/Datasets/flickrlogos_47_COCO/annotations/instances_train.json')
    #parser.add_argument('-feat_savedir', help='feature save directory', type=str, default='/home/jeancherubini/Documents/feature_maps')
    parser.add_argument('-principal_components', help='amount of components kept (depth of feature vectors)', type=int, default=64)   
    parser.add_argument('-model', help='model used for the convolutional features', type=str, choices=['resnet', 'VGG16', 'retinanet'], default='VGG16') 
    parser.add_argument('-layer', help='resnet layer(s) used for extraction, they can be:\n for VGG: {0}\n for resnet:{1}\n For multiple layers, a semicolon "," can be used to separate '.format(
    'conv1_relu, conv2_block3_out, conv3_block4_out, conv4_block6_out, conv5_block3_out',
    'block1_conv2, block2_conv2, block3_conv3, block4_conv3, block5_conv3'), type=str, default='block3_conv3') 
    parser.add_argument('-batch_size', help='size of the batch of features', type=int, default=3)
    parser.add_argument('-batches_pca', help='How many batches to se for PCA training', type=int, default=5)
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

    multiple_layers = params.layer.split(',')
    print('multiple_layers', multiple_layers)

    feature_getter = feature_getter_class()
    for layer in multiple_layers:
        params.layer = layer
        feature_getter.get_features(params)
