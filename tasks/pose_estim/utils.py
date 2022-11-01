import torch
import json
import os
import sys
import pprint
from glob import glob
import pickle
import numpy as np
from itertools import zip_longest
import io
from einops import rearrange, reduce, repeat
from PIL import Image, ImageDraw
import cv2
from collections import defaultdict

import albumentations as A
from albumentations.pytorch import transforms
# from albumentations.core.transforms_interface.Ima
import torch
 

# filename: bbox, class ...
mapper = defaultdict(list)
for ann in j['annotations']:
    image_id = ann['image_id']
    cat_id = ann['category_id']
    n_key = ann['num_keypoints']
    keypoints = ann['keypoints']
    bbox = ann['bbox']
    mapper[image_id].append({'n_key': n_key, 'keypoints': keypoints, 'bbox': bbox, 'cat_id': cat_id})
    break

skeleton = j['categories'][0]['skeleton']


def coco_to_keypoints():
    
    # "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"

    kp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 142, 309, 1, 177, 320, 2, 191, 398, 2, 237, 317, 2, 233, 426, 2, 306, 233, 2, 92, 452, 2, 123, 468, 2, 0, 0, 0, 251, 469, 2, 0, 0, 0, 162, 551, 2]
    ks = list()
    for i in range(0, len(kp), 3):
        x, y, v = kp[i:i+3]
        ks.append((x, y, v))

    image = Image.open('/mount/dataset/COCO/val2017/000000425226.jpg')

    return {'image': image, 'keypoints': ks}


def visualization():

    # "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"

    kp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 142, 309, 1, 177, 320, 2, 191, 398, 2, 237, 317, 2, 233, 426, 2, 306, 233, 2, 92, 452, 2, 123, 468, 2, 0, 0, 0, 251, 469, 2, 0, 0, 0, 162, 551, 2]
    ks = list()
    for i in range(0, len(kp), 3):
        x, y, v = kp[i:i+3]
        ks.append((x, y, v))

    image = Image.open('/mount/dataset/COCO/val2017/000000425226.jpg')
    draw = ImageDraw.Draw(image)

    for x, y, v in ks:
        if v > 0:
            draw.ellipse([x-5, y-5, x+5, y+5], fill=(255, 255, 0), outline ="red")

    for a, b in skeleton:
        if ks[a-1][-1] > 0 and ks[b-1][-1] > 0:
            draw.line((ks[a-1][:-1], ks[b-1][:-1]), fill=(255, 255, 0))

    image.show()