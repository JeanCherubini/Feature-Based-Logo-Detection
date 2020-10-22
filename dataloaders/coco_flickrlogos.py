import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader


class  Dataset_Flickrlogos_47(Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
    
    def __getitem__(self, index):
        #Own Coco File
        coco = self.coco
        #Image ID
        img_id = self.ids[index]
        #List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id) 
        #Dictionary: target coco_annotation file from an image
        coco_annotation = coco.loadAnns(ann_ids)
        #path for input_image
        path = coco.loadImgs(img_id)[0]['file_name']
        #open the input image
        img = Image.open(os.path.join(self.root, path)) 

        #number of objects in the image
        num_objs = len(coco_annotation)


        #Bounding boxes for objects
        #In coco format, bbox = [xmin, ymin, width, height]
        #In Pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][0]
            ymax = ymin + coco_annotation[i]['bbox'][4]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float64)

        #Labels (In my case, I only have one class: target or background)
        labels = torch.ones((num_objs), dtype = torch.int64)

        #Tensorise img
        img_id = troch.tensor([img_id])

        #Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)

        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)

# In my case, just added ToTensor
def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)

if __name__ == "__main__":
    #Path to own data train
    train_data_dir = "../../data/coco_flickrlogos_47/images/train"
    #Path to annotations
    train_coco = "../../data/coco_flickrlogos_47/annotations/instances_train.json"

    coco_dataset = Dataset_Flickrlogos_47(train_data_dir, train_coco, transforms = get_transform())

    #collate_fn needs for batch
    def collate_fn(batch):
        return tuple(zip(*batch))

    #batch size
    train_batch_size = 1

    #own dataloader
    data_loader = DataLoader(Dataset_Flickrlogos_47, train_batch_size, num_workers=2, collate_fn=collate_fn)


    # select device (whether GPU or CPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
