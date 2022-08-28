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


def converting_ubiai(data, write_file, labelss):
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


def is_on_same_line(box_a, box_b, min_y_overlap_ratio=0.8):
    """Check if two boxes are on the same line by their y-axis coordinates.

    Two boxes are on the same line if they overlap vertically, and the length
    of the overlapping line segment is greater than min_y_overlap_ratio * the
    height of either of the boxes.

    Args:
        box_a (list), box_b (list): Two bounding boxes to be checked
        min_y_overlap_ratio (float): The minimum vertical overlapping ratio
                                    allowed for boxes in the same line

    Returns:
        The bool flag indicating if they are on the same line
    """
    a_y_min = np.min(box_a[1::2])
    b_y_min = np.min(box_b[1::2])
    a_y_max = np.max(box_a[1::2])
    b_y_max = np.max(box_b[1::2])

    # Make sure that box a is always the box above another
    if a_y_min > b_y_min:
        a_y_min, b_y_min = b_y_min, a_y_min
        a_y_max, b_y_max = b_y_max, a_y_max

    if b_y_min <= a_y_max:
        if min_y_overlap_ratio is not None:
            sorted_y = sorted([b_y_min, b_y_max, a_y_max])
            overlap = sorted_y[1] - sorted_y[0]
            min_a_overlap = (a_y_max - a_y_min) * min_y_overlap_ratio
            min_b_overlap = (b_y_max - b_y_min) * min_y_overlap_ratio
            return overlap >= min_a_overlap or \
                overlap >= min_b_overlap
        else:
            return True
    return False


def stitch_boxes_into_lines(boxes, max_x_dist=10, min_y_overlap_ratio=0.8):
    """Stitch fragmented boxes of words into lines.

    Note: part of its logic is inspired by @Johndirr
    (https://github.com/faustomorales/keras-ocr/issues/22)

    Args:
        boxes (list): List of ocr results to be stitched
        max_x_dist (int): The maximum horizontal distance between the closest
                    edges of neighboring boxes in the same line
        min_y_overlap_ratio (float): The minimum vertical overlapping ratio
                    allowed for any pairs of neighboring boxes in the same line

    Returns:
        merged_boxes(list[dict]): List of merged boxes and texts
    """

    if len(boxes) <= 1:
        return boxes

    merged_boxes = []

    # sort groups based on the x_min coordinate of boxes
    x_sorted_boxes = sorted(boxes, key=lambda x: np.min(x['box'][::2]))
    # store indexes of boxes which are already parts of other lines
    skip_idxs = set()

    i = 0
    # locate lines of boxes starting from the leftmost one
    for i in range(len(x_sorted_boxes)):
        if i in skip_idxs:
            continue
        # the rightmost box in the current line
        rightmost_box_idx = i
        line = [rightmost_box_idx]
        for j in range(i + 1, len(x_sorted_boxes)):
            if j in skip_idxs:
                continue
            if is_on_same_line(x_sorted_boxes[rightmost_box_idx]['box'],
                               x_sorted_boxes[j]['box'], min_y_overlap_ratio):
                line.append(j)
                skip_idxs.add(j)
                rightmost_box_idx = j

        # split line into lines if the distance between two neighboring
        # sub-lines' is greater than max_x_dist
        lines = []
        line_idx = 0
        lines.append([line[0]])
        for k in range(1, len(line)):
            curr_box = x_sorted_boxes[line[k]]
            prev_box = x_sorted_boxes[line[k - 1]]
            dist = np.min(curr_box['box'][::2]) - np.max(prev_box['box'][::2])
            if dist > max_x_dist:
                line_idx += 1
                lines.append([])
            lines[line_idx].append(line[k])

        # Get merged boxes
        for box_group in lines:
            merged_box = {}
            merged_box['text'] = ' '.join(
                [x_sorted_boxes[idx]['text'] for idx in box_group])
            x_min, y_min = float('inf'), float('inf')
            x_max, y_max = float('-inf'), float('-inf')
            for idx in box_group:
                x_max = max(np.max(x_sorted_boxes[idx]['box'][::2]), x_max)
                x_min = min(np.min(x_sorted_boxes[idx]['box'][::2]), x_min)
                y_max = max(np.max(x_sorted_boxes[idx]['box'][1::2]), y_max)
                y_min = min(np.min(x_sorted_boxes[idx]['box'][1::2]), y_min)
            merged_box['box'] = [
                x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max
            ]
            merged_boxes.append(merged_box)

    return merged_boxes


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
        converting_ubiai(data, f, list_class)
    # slipt_wildreceipt()
    # get_class_list()