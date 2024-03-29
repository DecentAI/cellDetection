import os
import glob
from skimage import io
from torch.utils.data import Dataset
import numpy as np


class CellDataset(Dataset):

    def __init__(self,image_dir, mask_dir,  transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = glob.glob(image_dir+'/*.tif')
        self.masks = glob.glob(mask_dir+'/*.png')
        self.no_of_ims = len(self.images)


    def __len__(self):
        return self.no_of_ims
    
    def __getitem__(self, idx):
        if idx < self.no_of_ims:

            name = self.images[idx]
            image = np.array(io.imread(name),dtype=np.float32)
            mask_name = name.replace(".tif","_mask.png")
            mask_name = mask_name.replace("_images/","_images_masks/")
            mask = np.array(io.imread(mask_name),dtype=np.float32)

            mask[mask==255.0] = 1.0        
        else:
            print('set idx out of bound!')
            image = -1
            mask = -1 

        if self.transform is not None: 
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

