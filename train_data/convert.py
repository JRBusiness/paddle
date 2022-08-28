import json
import shutil
from glob import glob
import ast
import cv2
import numpy as np

from paddleocr import PaddleOCR
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
import math

class_index = {
    "PROVIDER_NAME": 1,
    "MEM_NAME": 2,
    "MEM_ID": 3,
    "GROUP_NUMBER": 4,
    "RXBIN": 5,
    "RXPCN": 6,
    "ISSUER_ID": 7,
    "RXGROUP": 8,
    "EFFECTIVE": 9,
    "DEPENDENTS": 10,
    "HEALTH_PLAN": 11,
    "DOB": 12,
    "PAYER_ID": 13,
    "COVERAGE_DATE": 14,
    "SUBCRIBER_ID": 15,
    "PCP": 16,
    "RXID": 17,
    "SUBSCRIBER_NAME": 19,
    "RX_PLAN": 20,
    "POLICY_NUMBER": 21,
}


def crop_img(image, polygon):
    top_left = tuple(int(val) for val in polygon[0])
    bottom_right = tuple(int(val) for val in polygon[2])
    return image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]


def converting_hasty(data, write_file):
    # data = json.load(open(file, 'r'))
    name = '.'.join(data['metadata']['name'].split('.')[-2:])
    new_data = {
        'file_name': name,
        'height': data['metadata']['height'],
        'width': data['metadata']['width'],
        'annotations': []
    }
    for item in data['instances']:

        ori_image = cv2.imread(file.replace('___objects.json', ''))
        instance_array = [item['points'][x:x + 2] for x in range(0, len(item['points']), 2)]
        cropped_image = crop_img(ori_image, instance_array)
        text = ocr.ocr(cropped_image, cls=True)
        if text:
            text = text[0][-1][0]
            labels = {
                'label': item['className'],
                'transcription': text,
                'points': item['points'],
            }
            new_data['annotations'].append(labels)
            print(text, end='')
    print(new_data['file_name'], end='')
    write_file.write(f'{new_data}\n')


def get_bbox(bbox):
    list_items = []
    for bb in bbox:
        # for i in ["x", "y"]:
        list_items.append([bb["x"], bb["y"]])
    return list_items


def get_bbox_from_list(bbox):
    return [[bbox[0], bbox[1]], [bbox[2], bbox[3]], [bbox[4], bbox[5]], [bbox[6], bbox[7]]]


def get_bbox_from_array(bbox):
    list_items = []
    for bb in bbox:
        for i in ["x", "y"]:
            list_items.append(bb[i])
    return list_items


def get_bbox_from_bbox(bbox):
    list_items = []
    for bb in bbox:
        for i in bb:
            list_items.append(i)
    return list_items


def merge_boxes(box1, box2):
    return [
        box1[0], box2[0],
        box1[1], box2[1],
        box1[2], box2[2],
        box1[3], box2[3]
    ]

def converting_ubiai2(data, write_file):
    final = []
    for line in data:
        name = line['documentName'].split('.jpg_')[0]
        if line['annotation']:
            line_result = [f'images_files/{name}.jpg', []]
            bboxs = line['annotation']
            seen = set()
            if bboxs:
                bbox_dict = {}
                for item in bboxs:
                    label = item['label']
                    if label not in ["NETWORK", "OUT_OF_POCKET", "RX_PLAN", "PEDIATRIC_MEMBER_DENTAL", "PLAN_CODE"]:
                        for box in item['boundingBoxes']:
                            bbox = box['normalizedVertices']
                            text = box['word']
                            if label in seen:
                                bbox_new = get_bbox_from_array(bbox)
                                bbox = merge_boxes(bbox_dict[label][1], bbox_new)
                                text = " ".join([bbox_dict[label][0], text])
                            if label not in seen:
                                seen.add(label)
                                for boxes in item['boundingBoxes']:
                                    bbox = boxes['normalizedVertices']
                                    bbox = get_bbox(bbox)
                            bbox_dict[label] = [text, bbox]
                for k, v in bbox_dict.items():
                    labels = {
                        'transcription': v[0],
                        'label': k,
                        'points': v[1],
                        "id":
                    }
                    line_result[1].append(labels)
                print(line_result)
                final.append(json.dumps(line_result))
    write_file.writelines(line for line in final)


def converting_ubiai(data, write_file):
    final = []
    for line in data:
        name = line['documentName'].split('.jpg_')[0]
        if line['annotation']:
            name = f'images_files/{name}.jpg'
            new_data = {
                'file_name': name,
                'height': line['tokens'][0]['height'],
                'width': line['tokens'][0]['width'],
                'annotations': []
            }
            bboxs = line['annotation']
            seen = set()
            if bboxs:
                bbox_dict = {}
                for item in bboxs:
                    label = item['label']
                    if label not in ["NETWORK", "OUT_OF_POCKET", "RX_PLAN", "PEDIATRIC_MEMBER_DENTAL", "PLAN_CODE"]:
                        label = class_index[label]
                        for box in item['boundingBoxes']:
                            bbox = box['normalizedVertices']
                            text = box['word']
                            if label in seen:
                                bbox_new = get_bbox_from_array(bbox)
                                bbox = merge_boxes(bbox_dict[label][1], bbox_new)
                                text = " ".join([bbox_dict[label][0], text])
                            if label not in seen:
                                seen.add(label)
                                for boxes in item['boundingBoxes']:
                                    bbox = boxes['normalizedVertices']
                                    bbox = get_bbox_from_array(bbox)
                            bbox_dict[label] = [text, bbox]
                for k, v in bbox_dict.items():
                    labels = {
                        'box': v[1],
                        'text': v[0],
                        'label': k,
                    }
                    new_data["annotations"].append(labels)
                print(new_data)
                final.append(json.dumps(new_data))
    write_file.writelines(line for line in final)


def slipt_wildreceipt():
    for i in ['train', 'test']:
        with open(f'wildreceipt/wildreceipt_{i}.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                file_name = line.split('", "')[0].split('": "')[-1]
                try:
                    shutil.move(f'wildreceipt/{file_name}', f'wildreceipt/{i}/')
                except Exception as e:
                    print(e)


def get_class_list():
    file = './train_data/wildreceipt/annotate.json'
    data = json.load(open(file, 'r'))
    labels = []
    index = 0
    seen = set()
    with open('./train_data/wildreceipt/class_list.txt', 'w') as f:
        for line in data:
            if line['annotation']:
                bboxs = line['annotation']
                if bboxs:
                    for item in bboxs:
                        label = item['label']
                        if label not in seen:
                            seen.add(label)
                            labels.append(f'{index}  {label}\n')
                            index += 1
        f.writelines([label for label in labels])

if __name__ == '__main__':
    # images_path = 'wildreceipt/'
    # ocr = PaddleOCR(use_angle_cls=True, lang='en')
    # cataglog = ['train', 'test']
    # for i in cataglog:
    #     with open(f'wildreceipt/openset_{i}.txt', 'w') as f:
    #         for file in glob(f'wildreceipt/{i}/*.json'):
    #             converting(file, f)
    file = './train_data/wildreceipt/annotate.json'
    data = json.load(open(file, 'r'))
    list_class = open('./train_data/wildreceipt/class_list.txt', 'r').readlines()
    with open(f'./train_data/wildreceipt/closeset_train.txt', 'w', encoding='utf-8') as f:
        converting_ubiai(data, f)
    # slipt_wildreceipt()
    # get_class_list()