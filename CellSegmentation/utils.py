import torch
import torchvision
from dataset import CellDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import cv2 

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
import PIL

from sklearn.metrics import average_precision_score
import torchvision.transforms as transforms

def save_checkpoint(state, filename="my_checkpoint_UNet3_large_epoch.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = CellDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CellDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def count_cells(pred_mask, t=65,name="test",thr=127,k_x=3,k_y=3):
    
    img_mask=np.squeeze(pred_mask)
    plt.imsave(f'test_{name}.png', img_mask)

    image = Image.open(f'test_{name}.png').convert("L")
    img = np.asarray(image)
    img = img.copy()
    

    blur = cv2.GaussianBlur(img, (k_x,k_y), 0)
    (t, binary) = cv2.threshold(blur, t, thr, cv2.THRESH_BINARY)

    (contours, _) = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # print("Found %d cells." % len(contours))

    return len(contours)


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0

    model.eval()
    
    num_total_cell = 0
    num_pred_cell = 0
    ap = 0 
    total_num = 0
    for x, y in loader:
        real = y
        
        ch,h,w = real.shape
        
        real = torch.unsqueeze(real, dim=0)
        real = F.interpolate(real, size=((h//32)*32, (w//32)*32), mode='bilinear', align_corners=True)
        real = torch.squeeze(real, dim=0)
        x = x.to(device).unsqueeze(1)
        y = real.to(device).unsqueeze(1)
        
        T = 10
        with torch.no_grad():
            preds = model(x)
            for t in range(T-1):
                preds=preds+model(x)
            preds = preds/T
        
        # for name, param in model.named_parameters(): 
        #     print("Name after training : ", name, "Param : ", param)
        #     break
        
        

        preds = (preds > 0.5).float()
        preds = F.interpolate(preds, size=((h//32)*32, (w//32)*32), mode='bilinear', align_corners=True)

        
        for i, pred in enumerate(preds): 
            
            torchvision.utils.save_image(pred, "pred.png")
            img = PIL.Image.open('pred.png').convert("L")
            tf = transforms.ToTensor()
            img_t = tf(img)
            img_t = (img_t > 0.5).float()
            img_t = img_t.to(device=device)
            
            ch_t,h_t,w_t = img_t.shape
            img_t = torch.unsqueeze(img_t, dim=0)
            img_t = F.interpolate(img_t, size=((h_t//32)*32, (w_t//32)*32), mode='bilinear', align_corners=True)
            img_t = torch.squeeze(img_t, dim=0)

            npy =  img_t.cpu().numpy()
            npy = np.squeeze(npy)

            
            num_pred = count_cells(cv2.cvtColor(npy,cv2.COLOR_GRAY2RGB),name="pred",thr=127,k_x=19,k_y=19)
            
            y_tmp = np.squeeze(y[i].cpu().numpy())
            num_y = count_cells(cv2.cvtColor(y_tmp,cv2.COLOR_GRAY2RGB),name="real",thr=127,k_x=1,k_y=1)
            

            num_total_cell += num_y
            num_pred_cell += num_pred
            

        


        num_correct += (preds == y).sum()
        num_pixels += torch.numel(preds)
        dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

        y_np = y.cpu().numpy().astype(int)
        preds_np = preds.cpu().numpy().astype(int)

        tp = ((y_np == 1) & (preds_np == 1)).sum()
        fp = ((y_np == 0) & (preds_np == 1)).sum()
        precision = (tp / (tp + fp))
        ap +=  precision/2
        total_num += 1

        # print("-----------Batch Performance------------------")
        # print(
        #     f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
        # )
        # print(
        #     f"mAP evaluation {ap/total_num*100:.2f}"
        # )
        # smaller = 0
        # bigger =0
        # if(num_pred_cell > num_total_cell):
        #     smaller = num_total_cell
        #     bigger = num_pred_cell
        # else:
        #     smaller = num_pred_cell
        #     bigger = num_total_cell
        # print(
        #     f"Cell counting accuracy in percentage {smaller/bigger*100:.2f}"
        # )
        # print(f"Dice score: {dice_score/len(loader)}")



    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(
        f"mAP evaluation {ap/total_num*100:.2f}"
    )
    smaller = 0
    bigger =0
    if(num_pred_cell > num_total_cell):
        smaller = num_total_cell
        bigger = num_pred_cell
    else:
        smaller = num_pred_cell
        bigger = num_total_cell
    print(
        f"Cell counting accuracy in percentage {smaller/bigger*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images_UNet3_uncertainty/", device="cuda"
):
    model.eval()
    counter = 0
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device).unsqueeze(1)
        with torch.no_grad():
            T = 10
            preds = torch.sigmoid(model(x))
            for t in range(T-1):
                preds=preds+torch.sigmoid(model(x))
            preds = preds/T
            
            
        preds = preds.squeeze()
        pred_fg = preds.cpu().numpy()
        pred_bg = 1-pred_fg
        
        preds = (preds > 0.5).float()
        U_ts = -(pred_fg*np.log(pred_fg)+pred_bg*np.log(pred_bg))

        for i in range(len(preds)):
            height,width = U_ts[i].shape
            bgr = cv2.cvtColor(x[i][0].cpu().numpy(), cv2.COLOR_GRAY2BGR)
            res = cv2.resize(bgr, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
            y_tmp = cv2.cvtColor(y[i].cpu().numpy(), cv2.COLOR_GRAY2BGR)
            y_tmp = y_tmp*255
            y_final = cv2.resize(y_tmp, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
            pd = cv2.cvtColor(preds[i].cpu().numpy(), cv2.COLOR_GRAY2BGR)*255
            uncertain = cv2.cvtColor(U_ts[i], cv2.COLOR_GRAY2BGR)*255

            h_img = cv2.hconcat([res, y_final,pd,uncertain])
            cv2.imwrite(f"{folder}{idx}_{i}_combined.png", h_img)
        if(counter == 100):
            break
        else:
            counter +=1
    model.train()


def save_predictions_default(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device).unsqueeze(1)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y, f"{folder}{idx}.png")

    model.train()