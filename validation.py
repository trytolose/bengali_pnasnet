import numpy as np
import pandas as pd
import cv2
from os.path import join
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import gc
import torch.utils.data as utils
from torch.utils.data import Dataset
from albumentations import *
from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize
from albumentations.pytorch import ToTensorV2, ToTensor
from tqdm import tqdm
import sklearn.metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
from radam import RAdam
import pretrainedmodels
from torch.utils.tensorboard import SummaryWriter
import sys
from collections import deque
import copy
import os
import argparse



class PretrainedCNN(nn.Module):
    def __init__(self):
        super(PretrainedCNN, self).__init__()
        self.model = pretrainedmodels.__dict__['pnasnet5large'](num_classes=1000, pretrained='imagenet')
        self.model.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        w = self.model.conv_0.conv.weight
        self.model.conv_0.conv = nn.Conv2d(1, 96, kernel_size=3, stride=2, bias=False)

        self.model.conv_0.conv.weight = nn.Parameter(torch.mean(w, dim=1, keepdim=True))
        self.model.last_linear = nn.Linear(4320, 186)        

    def forward(self, x):
        
        x = self.model(x)
        return x
    
    
class ImageDataset(Dataset):
    def __init__(self,
                 path,
                 transform,
                 df):
        self.path = path
        self.transform = transform
        self.df = df
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        
        img = cv2.imread(join(self.path, self.df['image_id'][idx]+".png"), -1)
        target = np.array([self.df['grapheme_root'][idx], self.df['vowel_diacritic'][idx], self.df['consonant_diacritic'][idx]])
        
        img = 255 - img
        img = np.expand_dims(img, axis=2)
        
        data = {"image": img}
        augmented = self.transform(**data)
        
        return augmented['image'], target
    
def predicts(model, val_loader, device, ext=False):
    
    val_true = []
    val_pred = []
    pred_grapheme = []
    pred_vowel = []
    pred_cons = []
    
    model.to(device)
    model.eval()  
    with torch.no_grad():
        for image, target in tqdm(val_loader, ncols=50):
            xs = image.to(device)
            ys = target.to(device)

            pred = model(xs)
            grapheme = pred[:,:168]
            vowel = pred[:,168:179]
            cons = pred[:,179:]

            grapheme = torch.softmax(grapheme, dim=1).cpu().data.numpy()
            vowel = torch.softmax(vowel, dim=1).cpu().data.numpy()
            cons = torch.softmax(cons, dim=1).cpu().data.numpy()
            val_true.append(target.numpy())
            pred_grapheme.append(grapheme)
            pred_vowel.append(vowel)
            pred_cons.append(cons)


    val_true = np.concatenate(val_true)
    pred_grapheme = np.concatenate(pred_grapheme)
    pred_vowel = np.concatenate(pred_vowel)
    pred_cons = np.concatenate(pred_cons)


    return val_true, pred_grapheme, pred_vowel, pred_cons
    
    
def valid(args):
    
    val_transform = Compose([
        Resize(224, 224),
        Normalize(
            mean=[0.0692],
            std=[0.205],
        ),
        ToTensorV2() 
    ])

    df_train = pd.read_csv("train_folds.csv")
    val = df_train[df_train['kfold']==0].reset_index(drop=True)#[:1000]
    val_data = ImageDataset('../input/images', val_transform, val)
    val_loader = utils.DataLoader(val_data, shuffle=False, num_workers=5, batch_size=32, pin_memory=True)   

    device = torch.device(f"cuda:{args.gpu_n}")
    
    model = PretrainedCNN()
    model.to(device)
    model.load_state_dict(torch.load(args.pretrain_path, map_location=f"cuda:{args.gpu_n}"))
    print("weights loaded")
    
    
    
    val_true, pred_grapheme, pred_vowel, pred_cons = predicts(model, val_loader, device)

    np.save("psa_grapheme.npy", pred_grapheme)
    np.save("psa_vowel.npy", pred_vowel)
    np.save("psa_cons.npy", pred_cons)
    
def main():
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--pretrain_path', type=str, default="", help='checkpoint name')
    parser.add_argument('--gpu_n', type=str, default="0", help='gpu cuda number')
    
    args = parser.parse_args()
    
    valid(args)
    
if __name__ == '__main__':
    main()