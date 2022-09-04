import json
import os
import shutil

import cv2

class_index = {
    "PROVIDER_NAME": 0,
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
    "POLICY_NUMBER": 1,
}

num_def = {
    "0": 51,
    "1": 2,
    "2": 3,
    "3": 4,
    "4": 5,
    "5": 6,
    "6": 7,
    "7": 8,
    "8": 9,
    "9": 10,
    "10": 11,
    "11": 12,
    "12": 13,
    "13": 14,
    "14": 16,
    "15": 17,
    "16": 18,
    "17": 19,
    "18": 20,
    "19": 21,
    "20": 22,
    "21": 23,
    "22": 24,
    "23": 25,
    "24": 26,
    "25": 27,
    "26": 28,
    "27": 29,
    "28": 30,
    "29": 31,
    "30": 32,
    "31": 33,
    "32": 34,
    "33": 35,
    "34": 36,
    "35": 37,
    "36": 38,
    "37": 39,
}
link_bbox = {
    "51": "21",
    "2": "22",
    "3": "23",
    "4": "24",
    "5": "25",
    "6": "26",
    "7": "27",
    "8": "28",
    "9": "29",
    "10": "30",
    "11": "31",
    "12": "32",
    "13": "33",
    "14": "34",
    "16": "36",
    "17": "37",
    "18": "38",
    "19": "39",
}
value_id = [
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
    "32",
    "33",
    "34",
    "36",
    "37",
    "38",
    "39",
]
def crop_img(image, polygon):
    top_left = tuple(int(val) for val in polygon[0])
    bottom_right = tuple(int(val) for val in polygon[2])
    return image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]


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
        [box1[0], box2[0]],
        [box1[1], box2[1]],
        [box1[2], box2[2]],
        [box1[3], box2[3]],
    ]


def converting_paddle_SDMGR(data, write_file):
    for line in data:
        name, annotation = line.split("\t")
        write_file.write(f"{name}\t")
        value_present = []
        annotation = json.loads(annotation)
        for item in annotation:
            for k, v in num_def.items():
                if str(item["key_cls"]) == str(v):
                    new_item = {
                        "transcription": item["transcription"],
                        "label": int(k),
                        "points": item["points"],
                    }
                    value_present.append(new_item)
        write_file.write(f"{json.dumps(value_present)}\n")


def converting_mmocr_SDMGR(data, write_file, class_list):
    for line in data:
        name, annotation = line.split("\t")
        annotation = json.loads(annotation)
        width, height, _ = cv2.imread(f"train_data/wildreceipt/{name}").shape
        new_data = {
            'file_name': name,
            'height': height,
            'width': width,
            'annotations': []
        }
        for item in annotation:
            for k, v in num_def.items():
                if item["id"] == v:
                    new_data["annotations"].append({
                        'box': get_bbox_from_bbox(item["points"]),
                        'text': item["transcription"],
                        'label': class_list[int(k)],
                    })
        write_file.write(json.dumps(new_data))
        write_file.write("\n")


def converting_paddle_SER(data1, data2, write_file):
    for line in data1:
        name, annotation = line.split("\t")
        write_file.write(f"{name}\t")
        value_present = []
        linking_box = link_bbox.items()
        annotation = json.loads(annotation)
        new_annotation = []
        for item in annotation:
            new_item = {
                "transcription": item["transcription"],
                "label": item["key_cls"],
                "points": item["points"],
                "id": 0,
                "linking": []
            }
            for a, b in [line2.split("\t") for line2 in data2]:
                if a == name:
                    for item2 in json.loads(b):
                        if new_item["transcription"] == item2["transcription"]:
                            key = item2["key_cls"]
                            new_item["id"] = int(key)
                            for k, v in linking_box:
                                if new_item["id"] == int(k):
                                    new_item["linking"] = [int(k), int(v)]
                                    value_present.append(int(v))
            new_annotation.append(new_item)
        for item in new_annotation:
            if item["id"] in value_present:
                for k, v in linking_box:
                    if int(v) == item["id"]:
                        item["linking"] = [int(k), int(v)]
        for item in new_annotation:
            if item["id"] in value_id and item["linking"] == []:
                item["label"] = "other"
                item["id"] = 1
        write_file.write(f"{json.dumps(new_annotation)}\n")


def change_label(data, writer):
    for line in data:
        name, annotation = line.split("\t")
        annotation = json.loads(annotation)
        collect = []
        writer.write(name + "\t")
        for item in annotation:

            if item["id"] in [0, 1, 2, 3, 4, 5, 6, 7, 8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]:
                collect.append(link_bbox[str(item["id"])])
                item["label"] = "question"
        for item in annotation:
            if item["id"] not in collect:
                item["label"] = "other"
            else:
                item["label"] = "answer"
            if item["id"] == 88:
                item["label"] = "ignore"
        writer.write(json.dumps(annotation))
        writer.write("\n")


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


def get_current():
    data1 = open("./train_data/wildreceipt/label1.txt", "r").readlines()
    data2 = open("./train_data/wildreceipt/label2.txt", "r").readlines()
    name2s = []
    for line2 in data2:
        name2s.append(line2.split("\t")[0])
    with open("./train_data/wildreceipt/labelnew.txt", "w") as f:
        lines = []
        for line in data1:
            name, _ = line.split("\t")
            if name in name2s:
                lines.append(line)
        f.writelines(lines)


if __name__ == '__main__':
    # file = './train_data/wildreceipt/annotate.json'
    # data = json.load(open(file, 'r'))
    # list_class = open('./train_data/wildreceipt/class_list.txt', 'r').readlines()
    # with open(f'./train_data/wildreceipt/paddle_sdgmr.txt', 'w', encoding='utf-8') as f:
    #     converting_paddle_SER(data, f)
        # converting_mmocr(data, f)
    data1 = open("./train_data/wildreceipt/labelnew.txt", "r").readlines()
    data2 = open("./train_data/wildreceipt/label2.txt", "r").readlines()
    data3 = open("./train_data/wildreceipt/paddle_ser.txt", "r").readlines()
    classlist = open("./train_data/wildreceipt/class", "r").readlines()
    with open(f'./train_data/wildreceipt/paddle_ser.txt', 'w', encoding='utf-8') as f:
        converting_paddle_SER(data1, data2, f)
    # get_current()