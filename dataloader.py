from torch.autograd import Variable
from torch.optim import Adam
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from torchvision.utils import make_grid
from torchvision.datasets.mnist import MNIST
from tqdm import tqdm
import torchnet as tnt
import cv2

BATCH_SIZE = 1
NUM_CLASSES = 5
NUM_EPOCHS = 200
NUM_ROUTING_ITERATIONS = 3

## Dataloader

import SimpleITK as sitk
import numpy as np
import torch, os, glob, tqdm
import matplotlib.pyplot as plt
%matplotlib inline


os.environ["CUDA_VISIBLE_DEVICES"]="2"


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from visdom import Visdom
from sklearn.model_selection import train_test_split

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = (path, class_to_idx[target])
                images.append(item)

    return images

def reader(path):
    img =sitk.GetArrayFromImage(sitk.ReadImage(path))
    #img = cv2.resize(img, (224,224))
    #img = np.expand_dims(img, axis=0)
    #img = np.concatenate([img,img,img], axis=0).transpose((1,2,0)).astype(np.float32)
    return img
def resize_img(img):
    gray_img = img.transpose((2,0,1))[0]
    resize_gray_img = cv2.resize(gray_img, (160,120))
    return resize_gray_img    

class DicomFolder(Dataset):
    def __init__(self, root, transform=None):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        
        if len(imgs)==0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"))
        
        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        
    def __getitem__(self, index):
        
        path, label = self.imgs[index]
        
        img = reader(path)
        resized_img = resize_img(img)
        #dicom=cv2.resize(dicom,(160,120))
       
        #dicom = np.expand_dims(cv2.resize(self.loader(path)[0], (224,224)), axis=2).astype(np.float32)
        
        return resized_img, label
        
        
    def __len__(self):
        return len(self.imgs)






train_dset = DicomFolder('/data2/ar/gestures20170903/01except/')
val_dset = DicomFolder('/data2/ar/gestures20170903/01/')
dsets = {'train':train_dset, 'val':val_dset}
dset_loaders = {x: DataLoader(dsets[x], batch_size=1,
                            shuffle=True, num_workers=4) for x in ['train', 'val']}
dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
dset_classes = dsets['train'].classes
dset_classes






































































































