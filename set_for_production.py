import os
import json
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path
from skimage import io, transform


def main():
    return 1

def create_instances_file(images_pool_dir, queries_dir, annotations_file):
    categories = []
    images = []
    annotations = []

    parent_dir = Path(images_pool_dir).parent
    annotations_folder = '{0}/annotations/'.format(parent_dir)
    if not os.path.isdir(annotations_folder):
        os.makedirs(annotations_folder)

    images_folder = '{0}/images/'.format(parent_dir)
    if not os.path.isdir(images_folder):
        os.makedirs(images_folder)


    print(images_folder)


    if annotations_file == '':
        annotations_object = {}
        query_classes = os.listdir(queries_dir)

        for id_, name in enumerate(query_classes):
            obj =  {'supercategories':0,'name':name, 'id':int(id_+1)}
            categories.append(obj)


        print(categories)

        images_list = os.listdir(images_pool_dir)
        id_image_counter = 0
        for image in images_list:

            path_to_img = '{0}/{1}'.format(images_pool_dir, image)

            #Obtener propiedades de la imagen
            img = mpimg.imread(path_to_img)
            assert(len(img.shape) == 3)
   
            resized_img = transform.rescale(img, [1/2,1/2,1], anti_aliasing=True)
            rotated_img = transform.rotate(resized_img,-90,resize=True)


            destino_imagen = '{0}/{1}'.format(images_folder, image)
            mpimg.imsave(destino_imagen, rotated_img)

            width = resized_img.shape[0]
            height = resized_img.shape[1]
            date_captured = datetime.now()
            #Almacenamiento de la informacion de la imagen
            images.append({"id":id_image_counter, "license":"", "width":width, "height":height, "file_name":image, "date_captured":str(date_captured)})
            id_image_counter += 1

        print(images)

    
    else: 
        #Por definir tratamiento para set de datos con anotaciones formato de anotaciones correspondiente para un archivo de texto, csv o json.
        annotations = []

    data={}
    data["images"] = images
    data["annotations"] = annotations 
    data["categories"] = categories
    print("Ready with instances file generation!")

    annotations_file_out =  '{0}/instances.json'.format(annotations_folder)
    with open(annotations_file_out, 'w') as outfile:
        json.dump(data, outfile)


    return 0


images_dir = '/home/jeancherubini/Downloads/isotonicas/mapa'
queries_dir = '/home/jeancherubini/Downloads/isotonicas/queries'  

create_instances_file(images_dir, queries_dir, '')

