import json
from glob import glob
import os
from PIL import Image, ImageDraw
import cv2
import numpy as np
import base64
from shapely.geometry import Polygon

def parse(j):
    '''
        json 파일 읽고 라벨 이름, 마스크 좌표 반환
    '''
    ret = None
    if isinstance(j, list):
        for item in j:
            parse(item)
    if isinstance(j, dict):
        for k, v in j.items():
            if 'shapes' in k:
                ret = v
                parse(v)
    return ret


def json_to_array(p):
    '''
        ndarray 학습 데이터를 만듬.
        학습할 때 dataset 에서 전처리할 거임
    '''

    # json parsing
    with open(p, 'r') as f:
        d = json.load(f)
    ret = parse(d)

    # fill masks
    p = p.replace('json', 'jpg')
    img = np.array(Image.open(p))
    final = np.zeros(shape=(np.array(img).shape[:2] + (3,)))
    
    for r in ret:
        points = r['points']
        fill = cv2.fillPoly(final.copy(), np.array([points]).astype(np.int32), (255, 255, 255))
    
        if r['label'] == 'hand':
            final[..., 0] = fill[..., 0]
        elif r['label'] == 'head':
            final[..., 1] = fill[..., 1]
        
        final[..., 2] = 255 - np.clip(final[..., 0] + final[..., 1], 0, 255)

    return final


def array_to_json(arr, path):
    '''
        prediction 결과를 json 형태로 바꿔 저장
        라벨링 툴에서 그대로 쓸 수 있게 함
    '''
    channel_mapper = {0: 'hand', 1: 'head', 2: 'background'}
    final = arr

    # create json skeleton
    j = dict()
    j['version'] = '5.1.1'
    j['flags'] = {}
    j['shapes'] = list()
    j['imagePath'] = path # 원본 이미지 path
    j['imageData'] = ''
    j['imageHeight'] = ''
    j['imageWidth'] = ''

    h, w = Image.open(path).size
    j['imageHeight'] = h
    j['imageWidth'] = w

    with open(path, 'rb') as img:
        base64_string = base64.b64encode(img.read())
    j['imageData'] = str(base64_string, 'utf-8')

    # convert array to points
    res = list()
    for ch in range(final.shape[-1]): # backgorund 제외
        conts, h = cv2.findContours(final[..., ch].astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        res.append(conts)

        # 모든 polygon 넓이 순으로 정렬하기 위한 작업
        areas = list()
        for cont in conts:
            epsilon1 = 0.005*cv2.arcLength(cont, True)
            approx1 = cv2.approxPolyDP(cont, epsilon1, True)
            approx1 = np.array(approx1).squeeze()
            
            if len(approx1) < 3:
                continue
            poly = Polygon(approx1)
            area = poly.area
            if area < 30 or not poly.is_valid:
                continue
            areas.append((area, approx1))

        # 넓이 순 정렬
        areas = sorted(areas, key=lambda x:x[0], reverse=True)

        # hand 면 제일 큰 거 2 개만 저장
        if ch == 0:
            polygons = areas[:2]
        # head 면 제일 큰 거 1 개만 저장
        elif ch == 1:
            polygons = areas[:1]
        else:
            pass
            
        for a, poly in polygons:
            dic = dict()
            dic['label'] = channel_mapper[ch]
            dic['points'] = poly.squeeze().tolist() #cont.squeeze()[::30, :].tolist()
            dic['group_id'] = None
            dic['shape_type'] = 'polygon'
            dic['flags'] = {}
            j['shapes'].append(dic)


############
        # for cont in conts:
        #     epsilon1 = 0.005*cv2.arcLength(cont, True)
        #     approx1 = cv2.approxPolyDP(cont, epsilon1, True)
        #     approx1 = np.array(approx1).squeeze()
            
        #     if len(approx1) < 3:
        #         continue
        #     poly = Polygon(approx1)
        #     if poly.area < 30 or not poly.is_valid:
        #         continue

        #     dic = dict()
        #     dic['label'] = channel_mapper[ch]
        #     dic['points'] = approx1.squeeze().tolist() #cont.squeeze()[::30, :].tolist()
        #     dic['group_id'] = None
        #     dic['shape_type'] = 'polygon'
        #     dic['flags'] = {}
            
        #     j['shapes'].append(dic)
            
    # save json
    # with open(path.replace('.jpg', '.json'), 'w') as f:
    #     json.dump(j, f)
    
    return j
