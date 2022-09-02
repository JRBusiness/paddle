import json
import shutil

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
    "0": 20,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "11": 11,
    "12": 12,
    "13": 13,
    "14": 14,
    "15": 16,
    "16": 17,
    "17": 18,
    "18": 21,
    "19": 22,
    "20": 23,
    "21": 24,
    "22": 25,
    "23": 26,
    "24": 27,
    "25": 28,
    "26": 29,
    "27": 30,
    "28": 31,
    "29": 32,
    "30": 33,
    "31": 34,
    "32": 36,
    "33": 37,
    "34": 38,
    "35": 88,
}
link_bbox = {
    "0": "20",
    "1": "21",
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


def converting_mmocr_SDMGR(data, write_file):
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
                "id": "",
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
        write_file.write(f"{json.dumps(new_annotation)}\n")


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
    data1 = open("./train_data/wildreceipt/train.txt", "r").readlines()
    data2 = open("./train_data/wildreceipt/label2.txt", "r").readlines()
    name2s = []
    for line2 in data2:
        name2s.append(line2.split("\t")[0])
    with open("./train_data/wildreceipt/image_files/label1.txt", "w") as f:
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
    data1 = open("./train_data/wildreceipt/label1.txt", "r").readlines()
    data2 = open("./train_data/wildreceipt/label2.txt", "r").readlines()
    with open(f'./train_data/wildreceipt/paddle_sdmgr.txt', 'w', encoding='utf-8') as f:
        converting_paddle_SDMGR(data2, f)
    # get_current()