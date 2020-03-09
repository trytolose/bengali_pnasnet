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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from radam import RAdam
import pretrainedmodels
import sys
from collections import deque
import copy
import os
from apex import amp
import argparse
import torch.multiprocessing




class PretrainedCNN(nn.Module):
    def __init__(self):
        super(PretrainedCNN, self).__init__()
        self.model = pretrainedmodels.__dict__['pnasnet5large'](num_classes=1000, pretrained=None)
        self.model.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
     #   w = self.model.conv_0.conv.weight
        self.model.conv_0.conv = nn.Conv2d(1, 96, kernel_size=3, stride=2, bias=False)

     #   self.model.conv_0.conv.weight = nn.Parameter(torch.mean(w, dim=1, keepdim=True))
        self.model.last_linear = nn.Linear(4320, 186)        

    def forward(self, x):
        
        x = self.model(x)
        return x
    
    
# Dataset class. Before training I unpack parquet and save pictures as .png
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



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#########################MIXUP/CUTMIX##########################################################################
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
def cutmix(data, targets1, targets2, targets3, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets1 = targets1[indices]
    shuffled_targets2 = targets2[indices]
    shuffled_targets3 = targets3[indices]

    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    targets = [targets1, shuffled_targets1, targets2, shuffled_targets2, targets3, shuffled_targets3, lam]
    return data, targets

def mixup(data, targets1, targets2, targets3, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets1 = targets1[indices]
    shuffled_targets2 = targets2[indices]
    shuffled_targets3 = targets3[indices]

    lam = np.random.beta(alpha, alpha)
    data = data * lam + shuffled_data * (1 - lam)
    targets = [targets1, shuffled_targets1, targets2, shuffled_targets2, targets3, shuffled_targets3, lam]

    return data, targets


def cutmix_criterion(preds1,preds2,preds3, targets):
    targets1, targets2,targets3, targets4,targets5, targets6, lam = targets[0], targets[1], targets[2], targets[3], targets[4], targets[5], targets[6]
    criterion = nn.CrossEntropyLoss(reduction='mean')
    return lam * criterion(preds1, targets1) + (1 - lam) * criterion(preds1, targets2) + \
        lam * criterion(preds2, targets3) + (1 - lam) * criterion(preds2, targets4) +   \
        lam * criterion(preds3, targets5) + (1 - lam) * criterion(preds3, targets6)

def mixup_criterion(preds1,preds2,preds3, targets):
    targets1, targets2,targets3, targets4,targets5, targets6, lam = targets[0], targets[1], targets[2], targets[3], targets[4], targets[5], targets[6]
    criterion = nn.CrossEntropyLoss(reduction='mean')
    return lam * criterion(preds1, targets1) + (1 - lam) * criterion(preds1, targets2) + \
        lam * criterion(preds2, targets3) + (1 - lam) * criterion(preds2, targets4) + \
        lam * criterion(preds3, targets5) + (1 - lam) * criterion(preds3, targets6)

##############################################################################################################

def train(args):
    # augmentations
    train_transform = Compose([

        Resize(args.img_size, args.img_size),
        Cutout(num_holes=8, max_h_size=20, max_w_size=20, fill_value=0, always_apply=False, p=0.5),
        Normalize(
                mean=[0.0692],
                std=[0.205],
            ),
        ToTensorV2()
    ])
    val_transform = Compose([
        Resize(args.img_size, args.img_size),
        Normalize(
            mean=[0.0692],
            std=[0.205],
        ),
        ToTensorV2() 
    ])

    
    # Load data
    df_train = pd.read_csv("../input/train_folds.csv")

    if args.fold == -1: 
        sys.exit()


    train = df_train[df_train['kfold']!=args.fold].reset_index(drop=True)#[:1000]
    val = df_train[df_train['kfold']==args.fold].reset_index(drop=True)#[:1000]

    train_data = ImageDataset('../input/images', train_transform, train)
    train_loader = utils.DataLoader(train_data, shuffle=True, num_workers=5, batch_size=args.batch_size, pin_memory=True)

    val_data = ImageDataset('../input/images', val_transform, val)
    val_loader = utils.DataLoader(val_data, shuffle=False, num_workers=5, batch_size=args.batch_size, pin_memory=True)   

# create model 

    device = torch.device(f"cuda:{args.gpu_n}")
    model = PretrainedCNN()
    
    
    if args.pretrain_path != "":
        model.load_state_dict(torch.load(args.pretrain_path, map_location=f"cuda:{args.gpu_n}"))
        print("weights loaded")
    model.to(device)
    
    
    
    optimizer = RAdam(model.parameters(), lr=args.start_lr)     

    opt_level = 'O1'
    model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', verbose=True, patience=8, factor=0.6)

    best_models = deque(maxlen=5)
    best_score = 0.99302


    for e in range(args.epoch):

        # Training:
        train_loss = []
        model.train()

        for image, target in tqdm(train_loader, ncols = 70):   
            optimizer.zero_grad()
            xs = image.to(device)
            ys = target.to(device)

            # Cutmix using with BUG
            if np.random.rand()<0.5:
                 images, targets = cutmix(xs, ys[:,0], ys[:,1], ys[:,2], 1.0)
                 pred = model(xs)
                 output1 = pred[:,:168]
                 output2 = pred[:,168:179]
                 output3 = pred[:,179:]
                 loss = cutmix_criterion(output1,output2,output3, targets)

            else:
                pred = model(xs)
                grapheme = pred[:,:168]
                vowel = pred[:,168:179]
                cons = pred[:,179:]

                loss = loss_fn(grapheme, ys[:,0]) + loss_fn(vowel, ys[:,1])+ loss_fn(cons, ys[:,2])

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            optimizer.step()
            train_loss.append(loss.item())

        
        #Validation    
        val_loss = []
        val_true = []
        val_pred = []
        model.eval()  
        with torch.no_grad():
            for image, target in val_loader:#tqdm(val_loader, ncols=50):
                xs = image.to(device)
                ys = target.to(device)

                pred = model(xs)
                grapheme = pred[:,:168]
                vowel = pred[:,168:179]
                cons = pred[:,179:]

                loss = loss_fn(grapheme, ys[:,0]) + loss_fn(vowel, ys[:,1])+ loss_fn(cons, ys[:,2])
                val_loss.append(loss.item())

                grapheme = grapheme.cpu().argmax(dim=1).data.numpy()
                vowel = vowel.cpu().argmax(dim=1).data.numpy()
                cons = cons.cpu().argmax(dim=1).data.numpy()
                val_true.append(target.numpy())
                val_pred.append(np.stack([grapheme, vowel, cons], axis=1))

        val_true = np.concatenate(val_true)
        val_pred = np.concatenate(val_pred)

        val_loss = np.mean(val_loss)
        train_loss = np.mean(train_loss)
        scores = []

        for i in [0,1,2]:
            scores.append(sklearn.metrics.recall_score(val_true[:,i], val_pred[:,i], average='macro'))
        final_score = np.average(scores, weights=[2,1,1])


        print(f'Epoch: {e:03d}; train_loss: {train_loss:.05f}; val_loss: {val_loss:.05f}; ', end='')
        print(f'score: {final_score:.5f} ', end='')

    
    #   Checkpoint model. If there are 2nd stage(224x224) save best 5 checkpoints
        if final_score > best_score:
            best_score = final_score
            state_dict = copy.deepcopy(model.state_dict()) 
            if args.save_queue==1:
                best_models.append(state_dict)
                for i, m in enumerate(best_models):
                    path = f"models/{args.exp_name}"
                    os.makedirs(path, exist_ok=True)
                    torch.save(m, join(path, f"{i}.pt"))
            else:
                path = f"models/{args.exp_name}"
                os.makedirs(path, exist_ok=True)
                torch.save(state_dict, join(path, "model.pt"))
            print('+')
        else:
            print()


        scheduler.step(final_score)
  
def main():
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_save_path', default="./", type=str, help='valid path')

    parser.add_argument('--fold', type=int, default=-1, help='fold')
    parser.add_argument('--encoder', type=str, default='resnet18', help='unet encoder')
    parser.add_argument('--epoch', type=int, default=100, help='total epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--start_lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--gpu_n', type=str, default="0", help='gpu cuda number')
    parser.add_argument('--exp_name', type=str, help='experiment name')
    parser.add_argument('--pretrain_path', type=str, default="", help='checkpoint name')
    parser.add_argument('--img_size', type=int, default=128, help='image size')
    parser.add_argument('--save_queue', type=int, default=0, help='image size')
    
    args = parser.parse_args()
    
    train(args)





if __name__ == '__main__':
    main()