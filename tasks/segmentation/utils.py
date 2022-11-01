import shutil
import json
from collections import defaultdict
import numpy as np
from PIL import Image
import os


def coco_to_segmentation():
    # 1. get raw label data
    annotation_path = '/mount/dataset/COCO/annotations/instances_val2017.json'
    with open(annotation_path, 'r') as f:
        d = json.load(f)

    # 2. parsing
    pair = defaultdict(list)
    for obj in d['annotations']:
        img_id = obj['image_id']
        category = obj['category_id']
        coords = obj['segmentation']
        pair[img_id].append((category, coords))
        break

    # 3. save image, label datas
    org_base = '/mount/dataset/COCO/val2017/'
    image_directory = '/mount/dataset/coco_convert/images/'
    label_directory = '/mount/dataset/coco_convert/labels'

    for img_id, objs in pair.items():

        # copy image
        img_id = ('0'*10+str(img_id))[-12:]
        org_path = os.path.join(org_base, f'{img_id}.jpg')
        img_path = os.path.join(image_directory, f'{img_id}.jpg')
        shutil.copy(org_path, img_path)
        
        # create label
        w, h, _ = np.array(Image.open(img_path)).shape

        for obj in objs:
            label = np.zeros(shape=(w, h, 1))

            # need fix.
            # 쪼개진 덩어리들 한 번에 그려주기
            # 라벨에 클래스 번호 값으로 채우기
            for coord in coords:
                # label = cv2.fillPoly(img=np.array(label), pts=coords_1, color=(255, 0, 0))
                pass
            # label = cv2.fillPoly(img=np.array(label), pts=coords_1, color=(255, 0, 0))
            # label = cv2.fillPoly(img=np.array(label), pts=coords_2, color=(255, 0, 0))

        # save label image
        np.save(os.path.join(label_directory, f'{img_id}'), label)

        break


def get_dice(pred, gt):

    # import matplotlib.pyplot as plt

    # x, y = np.meshgrid(range(1000), range(1000))
    # gt = np.sqrt(np.power(x-400, 2) + np.power(y-600, 2)) < 200
    # pred = np.sqrt(np.power(x-600, 2) + np.power(y-500, 2)) < 220

    union = np.sum(gt | pred)
    inter = np.sum(gt & pred)
    dice = 2 * inter / union
    
    return dice

