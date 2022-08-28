import json
import shutil
from glob import glob
import ast
import cv2
from paddleocr import PaddleOCR
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
import math

class_index = {
    'card': 1,
    'dob': 2,
    'rxgroup': 3,
    'plan': 4,
    'health_plan': 5,
    'mem_name': 6,
    'payer_id': 7,
    'dependents': 8,
    'mem_id': 9,
    'effective': 10,
    'coverage': 11,
    'subcriber_id': 12,
    'pcp': 13,
    'service_type': 14,
    'provider_name': 15,
    'rxbin': 16,
    'group_number': 17,
    'rxpcn': 18,
    'issuer': 19,
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
        for i in ["x", "y"]:
            list_items.append(bb[i])
    return list_items


def converting_ubiai(data, write_file, labelss):
    final = []
    for line in data:
        name = line['documentName'].split('.jpg_')[0]
        if line['annotation']:
            line_result = [f'images_files/{name}.jpg', []]
            bboxs = line['annotation']
            if bboxs:
                for item in bboxs:
                    for box in item['boundingBoxes']:
                        bbox = box['normalizedVertices']
                        text = box['word']
                        bbox = get_bbox(bbox)
                        labels = {
                            'label': item['label'],
                            'transcription': text,
                            'points': bbox,
                        }
                        line_result[1].append(labels)
                print(line_result)
                final.append(json.dumps(line_result))
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
    file = 'wildreceipt/annotate.json'
    data = json.load(open(file, 'r'))
    labels = []
    index = 0
    seen = set()
    with open('wildreceipt/class_list.txt', 'w') as f:
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
    file = 'wildreceipt/annotate.json'
    data = json.load(open(file, 'r'))
    list_class = open('wildreceipt/class_list.txt', 'r').readlines()
    with open(f'wildreceipt/closeset_train.txt', 'w', encoding='utf-8') as f:
        converting_ubiai(data, f, list_class)
    # slipt_wildreceipt()
    # get_class_list()