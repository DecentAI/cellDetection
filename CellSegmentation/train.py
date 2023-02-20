import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from unet import Unet,UNet3Plus

from dataset import CellDataset
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
<<<<<<< HEAD
import numpy as np
# hyperparams

LEARNING_RATE = 1E-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 50
# hyperparams

LEARNING_RATE = 1E-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 1

NUM_WORKERS = 8
IMAGE_HEIGHT = 520
IMAGE_WIDTH = 704
PIN_MEMORY = True
LOAD_MODEL = False
# TRAIN_IM_DIR = r'C:\Code\Dataset\LIVECell_dataset_2021\images\images\livecell_train_val_images'
# TRAIN_MASK_DIR = r'C:\Code\Dataset\LIVECell_dataset_2021\images\images\livecell_train_val_images_masks'
# VAL_IM_DIR = r'C:\Code\Dataset\LIVECell_dataset_2021\images\images\livecell_test_images'
# VAL_MASK_DIR = r'C:\Code\Dataset\LIVECell_dataset_2021\images\images\livecell_test_images_masks'

TRAIN_IM_DIR = r'/lunit/home/stevekang/decentAI/cellDetection/MCF7/LIVECell_dataset_2021/images/livecell_train_val_images'
TRAIN_MASK_DIR = r'/lunit/home/stevekang/decentAI/cellDetection/MCF7/LIVECell_dataset_2021/images/livecell_train_val_images_masks'
VAL_IM_DIR = r'/lunit/home/stevekang/decentAI/cellDetection/MCF7/LIVECell_dataset_2021/images/livecell_test_images'
VAL_MASK_DIR = r'/lunit/home/stevekang/decentAI/cellDetection/MCF7/LIVECell_dataset_2021/images/livecell_test_images_masks'



def train_model(loader, model, optimizer, loss_fcn, scaler):    
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loader):
        data = data.to(device=DEVICE).unsqueeze(1)
        targets = targets.to(device=DEVICE).unsqueeze(1)
        # forward path through model

        with torch.cuda.amp.autocast():
            pred = model(data)

            _,ch,h,w = targets.shape
            targets = F.interpolate(targets, size=((h//32)*32, (w//32)*32), mode='bilinear', align_corners=True)


            loss = loss_fcn(pred,targets)
        # backward path
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tdqm status

        loop.set_postfix(loss= loss.item())


def main():

    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    train_loader, val_loader = get_loaders(
        TRAIN_IM_DIR,
        TRAIN_MASK_DIR,
        VAL_IM_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        None, # disable data augmentation
        None, # disable data augmentation
        NUM_WORKERS,
        PIN_MEMORY,
    )

    model = UNet3Plus(in_channels=1, n_classes=1).to(DEVICE)
    model= nn.DataParallel(model)
    model.to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


    if LOAD_MODEL:

        load_checkpoint(torch.load("my_checkpoint_UNet3.pth.tar"), model)


    # check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()
    # save_predictions_as_imgs(
    #         val_loader, model, folder="saved_images_UNet3_uncertainty/", device=DEVICE
    #     )
    

    for epoch in range(NUM_EPOCHS):
        train_model(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }

        save_checkpoint(checkpoint, filename=f"UNet3_large_epoch_{epoch}.pth.tar")

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        # save_predictions_as_imgs(

        #     val_loader, model, folder="saved_images_UNet3/", device=DEVICE
        # )

    

if __name__ == "__main__":
    main()