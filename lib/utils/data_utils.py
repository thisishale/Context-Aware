import numpy as np
import random
import pickle as pkl
import os
import copy
import torch
import torch.utils.data as data
from lib.dataloaders import build_dataset

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def build_data_loader(args, phase='train',batch_size=None, scaler_sp=None):
    shuffle=phase=='train'
    print("shuffle is ",shuffle)
    dataset = build_dataset(args, phase, scaler_sp)
    scaler_sp = dataset.scaler_sp
    data_loaders = data.DataLoader(
        dataset,
        batch_size=args.batch_size if batch_size is None else batch_size,
        shuffle=phase=='train',
        num_workers=args.num_workers)

    return data_loaders, scaler_sp


def cxcywh_to_x1y1x2y2(boxes):
    '''
    Params:
        boxes:(Cx, Cy, w, h)
    Returns:
        (x1, y1, x2, y2 or tlbr
    '''
    new_boxes = np.zeros_like(boxes)
    new_boxes[...,0] = boxes[...,0] - boxes[...,2]/2
    new_boxes[...,1] = boxes[...,1] - boxes[...,3]/2
    new_boxes[...,2] = boxes[...,0] + boxes[...,2]/2
    new_boxes[...,3] = boxes[...,1] + boxes[...,3]/2
    return new_boxes


def bbox_normalize(bbox,W=1280,H=640):
    '''
    normalize bbox value to [0,1]
    :Params:
        bbox: [cx, cy, w, h] with size (times, 4), value from 0 to W or H
    :Return:
        bbox: [cx, cy, w, h] with size (times, 4), value from 0 to 1
    '''
    new_bbox = copy.deepcopy(bbox)
    new_bbox[:,0] /= W
    new_bbox[:,1] /= H
    new_bbox[:,2] /= W
    new_bbox[:,3] /= H
    
    return new_bbox

def bbox_denormalize(bbox,W=1280,H=640):
    '''
    normalize bbox value to [0,1]
    :Params:
        bbox: [cx, cy, w, h] with size (times, 4), value from 0 to 1
    :Return:
        bbox: [cx, cy, w, h] with size (times, 4), value from 0 to W or H
    '''
    new_bbox = copy.deepcopy(bbox)
    new_bbox[..., 0] *= W
    new_bbox[..., 1] *= H
    new_bbox[..., 2] *= W
    new_bbox[..., 3] *= H
    
    return new_bbox







