# based on https://github.com/zijundeng/pytorch-semantic-segmentation/blob/master/datasets/voc.py
import os

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils import data
from .base_dataset import BaseDataset

'''
color map
0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle # 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable,
12=dog, 13=horse, 14=motorbike, 15=person # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
'''
palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

class VOC(BaseDataset):
    def __init__(self, 
                 root, 
                 mode, 
                 num_classes=21,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=255, 
                 base_size=400, 
                 crop_size=(256, 256),
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4):

        super().__init__(ignore_label, base_size,
                crop_size, scale_factor, mean, std)
        self.imgs = make_dataset(root, mode)

        self.mode = mode
        self.root = root
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip

        self.ignore_label = ignore_label
        
        self.color_list = [[0, 128, 192], [128, 0, 0], [64, 0, 128],
                             [192, 192, 128], [64, 64, 128], [64, 64, 0],
                             [128, 64, 128], [0, 0, 192], [192, 128, 128],
                             [128, 128, 128], [128, 128, 0]]
        
        self.class_weights = None
        
        self.bd_dilate_size = bd_dilate_size

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = Image.open(os.path.join(self.root,'camvid',item["img"])).convert('RGB')
        image = np.array(image)
        size = image.shape

        color_map = Image.open(os.path.join(self.root,'camvid',item["label"])).convert('RGB')
        color_map = np.array(color_map)
        label = self.color2label(color_map)

        image, label, edge = self.gen_sample(image, label, 
                                self.multi_scale, self.flip, edge_pad=False,
                                edge_size=self.bd_dilate_size, city=False)

        return image.copy(), label.copy(), edge.copy(), np.array(size), name
    
    def __getitem__(self, index):
        if self.mode == 'test':
            # img_path, img_name = self.imgs[index]
            # img = Image.open(os.path.join(img_path, img_name + '.jpg')).convert('RGB')
            # if self.transform is not None:
            #     img = self.transform(img)
            # return img_name, img
            raise NotImplementedError

        img_path, mask_path = self.imgs[index]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        size = image.shape

        if self.mode == 'train':
            mask = sio.loadmat(mask_path)['GTcls']['Segmentation'][0][0]
            mask = Image.fromarray(mask.astype(np.uint8))
        else:
            mask = Image.open(mask_path)

        mask = np.array(mask)

        image, label, edge = self.gen_sample(image, mask, 
                                self.multi_scale, self.flip, edge_pad=False,
                                edge_size=self.bd_dilate_size, city=False)

        
        return image.copy(), label.copy(), edge.copy(), np.array(size), img_path

    def __len__(self):
        return len(self.imgs)
   

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


def make_dataset(root, mode):
    assert mode in ['train', 'val', 'test']
    items = []
    if mode == 'train':
        img_path = os.path.join(root, 'benchmark_RELEASE', 'dataset', 'img')
        mask_path = os.path.join(root, 'benchmark_RELEASE', 'dataset', 'cls')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'benchmark_RELEASE', 'dataset', 'train.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.mat'))
            items.append(item)
    elif mode == 'val':
        img_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'JPEGImages')
        mask_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'SegmentationClass')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Segmentation', 'seg11valid.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
            items.append(item)
    else:
        img_path = os.path.join(root, 'VOCdevkit (test)', 'VOC2012', 'JPEGImages')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'VOCdevkit (test)', 'VOC2012', 'ImageSets', 'Segmentation', 'test.txt')).readlines()]
        for it in data_list:
            items.append((img_path, it))
    return items

