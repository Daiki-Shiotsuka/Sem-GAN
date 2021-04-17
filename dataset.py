import os
import sys
import glob
import random

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np

class CycleGANDataset(Dataset):

    def __init__(self, dataset_name='day2night', transform=None, crop_size=(600,600), resize_size=(256,256),seg_channels=None, mode='train',p=0.2):

        self.transform = transform
        self.mode = mode
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.seg_channels  = seg_channels
        self.p = p # rate of segmentation dropout

        dataset_dir = os.path.join('dataset', dataset_name)

        if self.mode == 'train':
            self.imgs_A = sorted(glob.glob(os.path.join(dataset_dir, f"{mode}A", "imagesA/*.*")))
            self.imgs_B = sorted(glob.glob(os.path.join(dataset_dir, f"{mode}B", "imagesB/*.*")))
            self.labels_A = sorted(glob.glob(os.path.join(dataset_dir, f"{mode}A", "labelsA/*.*")))
            self.labels_B = sorted(glob.glob(os.path.join(dataset_dir, f"{mode}B", "labelsB/*.*")))
        else:
            self.imgs_A = sorted(glob.glob(os.path.join(dataset_dir, f"{mode}A", "*.*")))
            self.imgs_B = sorted(glob.glob(os.path.join(dataset_dir, f"{mode}B", "*.*")))

    def __getitem__(self, index):

        # load images
        item_A = cv2.imread(self.imgs_A[index])

        if self.mode == 'train':
            label_A = cv2.imread(self.labels_A[index])
            idx_B = random.randint(0, len(self.imgs_B)-1)
            item_B = cv2.imread(self.imgs_B[idx_B])
            label_B = cv2.imread(self.labels_B[idx_B])
        elif self.mode == 'val':
            idx_B = random.randint(0, len(self.imgs_B)-1)
            item_B = cv2.imread(self.imgs_B[idx_B])
        else: #self.mode = 'test'
            item_B = cv2.imread(self.imgs_B[index])

        # BGR -> RGB
        item_A = cv2.cvtColor(item_A, cv2.COLOR_BGR2RGB)
        item_B = cv2.cvtColor(item_B, cv2.COLOR_BGR2RGB)


        # crop
        h, w, _ = item_A.shape
        if self.crop_size:
            if self.mode == 'train':# or self.mode == 'val':
                new_h, new_w = self.crop_size
                top   = random.randint(0, h - new_h)
                left  = random.randint(0, w - new_w)
                item_A  = item_A[top:top + new_h, left:left + new_w]
                item_B  = item_B[top:top + new_h, left:left + new_w]
                label_A = label_A[top:top + new_h, left:left + new_w]
                label_B = label_B[top:top + new_h, left:left + new_w]
            else :
                new_h, new_w = self.crop_size
                new_h, new_w = int(new_h/2), int(new_w/2)
                test_w_left, test_w_right = int(w/2-new_h), int(w/2+new_h)
                test_h_left, test_h_right = int(h/2-new_w), int(h/2+new_w)
                item_A   = item_A[test_h_left:test_h_right, test_w_left:test_w_right]
                item_B   = item_B[test_h_left:test_h_right, test_w_left:test_w_right]

        # resize
        item_A  = cv2.resize(item_A, self.resize_size, cv2.INTER_LINEAR)
        item_B  = cv2.resize(item_B, self.resize_size, cv2.INTER_LINEAR)

        # transform -preprocessing
        item_A = self.transform(item_A)
        item_B = self.transform(item_B)

        if self.mode == 'train':
            label_A = cv2.resize(label_A, self.resize_size, cv2.INTER_NEAREST)
            label_B = cv2.resize(label_B, self.resize_size, cv2.INTER_NEAREST)
            label_A = cv2.cvtColor(label_A, cv2.COLOR_BGR2GRAY)
            label_B = cv2.cvtColor(label_B, cv2.COLOR_BGR2GRAY)
            label_A = torch.from_numpy(label_A.copy()).long()
            label_B = torch.from_numpy(label_B.copy()).long()
            label_A[label_A==255] = 19 # label 255 is unknown in BDD dataset
            label_B[label_B==255] = 19

            if  np.random.uniform(0,1) <= self.p:
                mask_A = label_A
                mask_B = label_B
                if label_A.unique().size() > label_B.unique().size():
                    tmp1, tmp2 = label_A.unique(), label_B.unique()
                else:
                    tmp1, tmp2 = label_B.unique(), label_A.unique()
                label_ab = []
                for i in tmp1:
                    if i in tmp2:
                        label_ab.append(i.item())

                for i in range(20):
                    if i in label_ab:
                        pass
                    else:
                        label_A[label_A==i] = 19
                        label_B[label_B==i] = 19
                        mask_A[label_A==i] = 0
                        mask_B[label_B==i] = 0

                mask_A = mask_A.repeat(3, 1, 1)
                mask_B = mask_B.repeat(3, 1, 1)
                item_A = item_A*mask_A
                item_B = item_B*mask_B

            h, w = label_A.size()
            target_A = torch.zeros(self.seg_channels, h, w) #self.seg_channels=20 (BDD dataset has 20 labels)
            target_B = torch.zeros(self.seg_channels, h, w)
            for c in range(self.seg_channels):
                target_A[c][label_A == c] = 1
                target_B[c][label_B == c] = 1
            return {'A': item_A, 'B': item_B, 'lA': target_A, 'lB': target_B}

        else:
            return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.imgs_A), len(self.imgs_B))
