import shutil
import json
from collections import defaultdict
from collections import Counter


class Bbox:
    def __init__(self, coord, coord_type='cxy'):
        # type: cxy, xywh, tlbr
        self.x1, self.x2, self.y1, self.y2 = coord

    def get_area(self):
        area = (self.x2 - self.x1) * (self.y2 - self.y1)
        return area



def coco_to_detection():
    # 1. get raw label data
    annotation_path = '/mount/dataset/COCO/annotations/instances_val2017.json'
    with open(annotation_path, 'r') as f:
        d = json.load(f)

    # 2. parsing
    pair = defaultdict(list)
    for obj in d['annotations']:
        img_id = obj['image_id']
        bbox = obj['bbox']
        category = obj['category_id']
        pair[img_id].append((category, bbox))

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
        ss = ''
        for obj in objs:
            a, b = obj
            ss += (str((a, *b))[1:-1]+'\n')

        with open(os.path.join(label_directory, f'{img_id}.txt'), 'w') as f:
            f.write(ss)

        break


def get_iou(bbox_1, bbox_2):
    '''
        format: (x1, x2, y1, y2)
    '''
    
    min_x_1, max_x_1, min_y_1, max_y_1 = bbox_1
    min_x_2, max_x_2, min_y_2, max_y_2 = bbox_2

    area_1, area_2 = (max_x_1 - min_x_1) * (max_y_1 - min_y_1), \
            (max_x_2 - min_x_2) * (max_y_2 - min_y_2)

    if area_1 <= 0 or area_2 <= 0:
        return 0
    
    wrap_x, wrap_y = 0, 0

    # box 2 가 box 1 의 완전 오른쪽 이거나 box 1 이 box 2 완전 오른쪽
    if max_x_1 < min_x_2 or max_x_2 < min_x_1:
        wrap_x = 0
    # box 2 가 box 1 안에 완전 포함됨
    elif min_x_1 <= min_x_2 < max_x_2 <= max_x_1:
        wrap_x = max_x_2 - min_x_2
    # 겹침 - box 1 < box 2
    elif min_x_1 < min_x_2 < max_x_1 < max_x_2:
        wrap_x = max_x_1 - min_x_2
    # box 1 이 box 2 안에 완전 포함됨
    elif min_x_2 <= min_x_1 < max_x_1 <= max_x_2:
        wrap_x = max_x_1 - min_x_1
    # 겹침 2 - box 2 < box 1
    elif min_x_2 < min_x_1 < max_x_2 < max_x_1:
        wrap_x = max_x_2 - min_x_1

    # box 2 가 box 1 의 완전 오른쪽 이거나 box 1 이 box 2 완전 오른쪽
    if max_y_1 < min_y_2 or max_y_2 < min_y_1:
        wrap_y = 0
    # box 2 가 box 1 안에 완전 포함됨
    elif min_y_1 <= min_y_2 < max_y_2 <= max_y_1:
        wrap_y = max_y_2 - min_y_2
    # 겹침 - box 1 < box 2
    elif min_y_1 < min_y_2 < max_y_1 < max_y_2:
        wrap_y = max_y_1 - min_y_2
    # box 1 이 box 2 안에 완전 포함됨
    elif min_y_2 <= min_y_1 < max_y_1 <= max_y_2:
        wrap_y = max_y_1 - min_y_1
    # 겹침 2 - box 2 < box 1
    elif min_y_2 < min_y_1 < max_y_2 < max_y_1:
        wrap_y = max_y_2 - min_y_1

    intersection = wrap_x * wrap_y
    union = area_1 + area_2 - intersection
    iou = intersection / union

    return iou



# NMS
# bboxs = \
#     [
#         # class, score, x1, x2, y1, y2
#         # class: 0~2
#         [0, .9, .2, .4, .2, .4],
#         [0, .8, .3, .5, .2, .4],
#         [0, .9, .2, .4, .2, .4],
#         [0, .3, .2, .4, .2, .4],
#         [1, .8, .5, .6, .7, .9],
#         [1, .8, .5, .6, .7, .9],
#         [1, .7, .5, .6, .7, .9],
#         [2, .8, .2, .4, .5, .7],
#         [2, .2, .1, .7, .2, .9],
#     ]

def NMS(bboxs):

    final_bboxs = list()
    thresh = .5
    thresh_iou = .5
    bboxs = [bbox for bbox in bboxs if bbox[1] > thresh]
    bboxs = sorted(bboxs, key=lambda x:x[1], reverse=True)

    while bboxs:
        selected_bbox = bboxs.pop(0)
        bboxs = [bbox for bbox in bboxs if bbox[0]!=selected_bbox[0] \
                or get_iou(bbox[2:], selected_bbox[2:]) < thresh_iou]
        
        final_bboxs.append(selected_bbox)

    return final_bboxs


def MeanAP(pred, gt):
    classes = [0, 1, 2]
    thresh_iou = .5
    ap = list()

    for cls in classes: # per class
        pred_cls = list()
        gt_cls = list()

        for p in pred:
            if p[1] == cls:
                pred_cls.append(p)

        for g in gt:
            if g[1] == cls:
                gt_cls.append(g)

        amounts = Counter([g[0] for g in gt_cls])

        for k, v in amounts.items():
            amounts[k] = np.zeros(v)


        pred_cls.sort(key=lambda x:x[2], reverse=True)

        TP = np.zeros(len(pred_cls))
        FP = np.zeros(len(pred_cls))
        total_true = len(gt_cls)

        for det_index, det in enumerate(pred_cls):
            gt_img = [bbox for bbox in gt_cls if bbox[0]==det[0]]
            num_gts = len(gt_img)
            best_iou = 0

            for idx, g in enumerate(gt_img):
                iou = get_iou(det[3:], g[3:])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            
            if best_iou > thresh_iou:
                if amounts[det[0]][best_idx] == 0:
                    TP[det_index] = 1
                    amounts[det[0]][best_idx] == 1
                else:
                    FP[det_index] = 1
            else:
                FP[det_index] = 1

        TP_sum = np.cumsum(TP, axis=0)
        FP_sum = np.cumsum(FP, axis=0)

        r = TP / (total_true)
        p = TP / (TP_sum + FP_sum)
        p = np.concatenate((np.array([1]), p))
        r = np.concatenate((np.array([0]), r))

        ap.append(np.trapz(p, r))
    mAP = sum(ap) / len(ap)
    
    return mAP