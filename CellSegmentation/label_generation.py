import numpy as np
import pandas as pd 
from skimage import io
from skimage.draw import rectangle_perimeter, polygon
import glob as glob
import json
import os

from tqdm import tqdm


dir_to_test_ims = r'decentAI/cellDetection/MCF7/LIVECell_dataset_2021/images/livecell_test_images'
dir_to_train_ims = r'decentAI/cellDetection/MCF7/LIVECell_dataset_2021/images/livecell_train_val_images'
dir_to_val_labels = r'decentAI/cellDetection/MCF7/LIVECell_dataset_2021/annotations/LIVECell_single_cells/mcf7/val.json'
dir_to_train_labels = r'decentAI/cellDetection/MCF7/LIVECell_dataset_2021/annotations/LIVECell_single_cells/mcf7/train.json'
dir_to_test_labels = r'decentAI/cellDetection/MCF7/LIVECell_dataset_2021/annotations/LIVECell_single_cells/mcf7/test.json'


# dir_to_train_labels = r'C:\Code\Dataset\LIVECell_dataset_2021\annotations\LIVECell\livecell_coco_train.json'
# dir_to_test_labels = r'C:\Code\Dataset\LIVECell_dataset_2021\annotations\LIVECell\livecell_coco_test.json'

def get_image_id_by_name(name, coco_val_labels, coco_train_labels):
    id = -1
    val_test_flag = -1
    for i in range(len(coco_val_labels['images'])):
        if coco_val_labels['images'][i]['file_name'] == name:
            id = coco_val_labels['images'][i]['id']
            val_test_flag = 0
              

    for i in range(len(coco_train_labels['images'])):
        if coco_train_labels['images'][i]['file_name'] == name:
            id = coco_train_labels['images'][i]['id']
            val_test_flag = 1
            
    return id, val_test_flag

def get_annotations_by_id(id, train_val_flag, coco_val_labels, coco_train_labels):
    annotations = [] # list to stack annotations per image 
    if train_val_flag == 0: # annotations in validation set
        for i in range(len(coco_val_labels['annotations'])):
            if coco_val_labels['annotations'][i]['image_id'] == id:
                annotations.append([coco_val_labels['annotations'][i]['id'],coco_val_labels['annotations'][i]['category_id'], coco_val_labels['annotations'][i]['segmentation'], coco_val_labels['annotations'][i]['bbox']])
    if train_val_flag == 1: # annotations in validation set
        for i in range(len(coco_train_labels['annotations'])):
            if coco_train_labels['annotations'][i]['image_id'] == id:
                annotations.append([coco_train_labels['annotations'][i]['id'],coco_train_labels['annotations'][i]['category_id'], coco_train_labels['annotations'][i]['segmentation'], coco_train_labels['annotations'][i]['bbox']])
    
    return annotations

def draw_bounding_boxes(im, annotation):
    local_im = np.copy(im) # ensure that the orig. im is untouched 
    for i in range(len(annotation)):
        bbox_corner = (int(annotation[i][3][1]),int(annotation[i][3][0]))
        bbox_extend = (int(annotation[i][3][3]),int(annotation[i][3][2]))
        rr,cc = rectangle_perimeter(bbox_corner, extent=bbox_extend, shape=local_im.shape)
        local_im[rr, cc] = 255
    
    return local_im

def generate_segmentation_mask(im, annotation):
    local_im = np.zeros_like(im,dtype=np.uint8)
    for i in range(len(annotation)):
        r = np.zeros(int(len(annotation[i][2][0])/2))
        c = np.zeros(int(len(annotation[i][2][0])/2))
        for j in range(int(len(annotation[i][2][0])/2)):
           # local_im[int(annotations[i][2][0][j*2+1])-1, int(annotations[i][2][0][j*2])-1] = 255
            r[j] = int(annotation[i][2][0][j*2+1])-1
            c[j] = int(annotation[i][2][0][j*2])-1
        rr, cc = polygon(r, c)
        local_im[rr, cc] = 255

    return local_im



# # create folder for label masks
# dir_to_masks = dir_to_train_ims + '_masks'
# if not os.path.isdir(dir_to_masks):
#     os.makedirs(dir_to_masks)

# # load image file paths
# ims = glob.glob(dir_to_train_ims + '/*.tif')

# # load label json object 
# with open(dir_to_val_labels) as f:
#     coco_val_labels = json.load(f)

# with open(dir_to_train_labels) as f:
#     coco_train_labels = json.load(f)

# with open(dir_to_test_labels) as f:
#     coco_test_labels = json.load(f)

# # generate label masks
# for image in ims:
#     imname = os.path.split(image)[-1]
#     im = io.imread(image)
#     single_label, train_val_flag = get_image_id_by_name(imname, coco_val_labels, coco_train_labels)
#     annotations = get_annotations_by_id(single_label, train_val_flag, coco_val_labels, coco_train_labels)
#     segmentation_mask = generate_segmentation_mask(im, annotations)
#     io.imsave(dir_to_masks+'/'+ imname[:-4]+'_mask.png', segmentation_mask)





# load label json object 
with open(dir_to_val_labels) as f:
    coco_val_labels = json.load(f)

with open(dir_to_train_labels) as f:
    coco_train_labels = json.load(f)

with open(dir_to_test_labels) as f:
    coco_test_labels = json.load(f)


dir_to_masks = dir_to_train_ims + '_masks'
if not os.path.isdir(dir_to_masks):
    os.makedirs(dir_to_masks)

# load image file paths
ims = glob.glob(dir_to_train_ims + '/*.tif')

for image in tqdm(ims):
    imname = os.path.split(image)[-1]
    im = io.imread(image)
    single_label, train_val_flag = get_image_id_by_name(imname, coco_val_labels, coco_train_labels)
    annotations = get_annotations_by_id(single_label, train_val_flag, coco_val_labels, coco_train_labels)

    segmentation_mask = generate_segmentation_mask(im, annotations)
    io.imsave(dir_to_masks+'/'+ imname[:-4]+'_mask.png', segmentation_mask)