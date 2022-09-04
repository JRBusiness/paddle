import base64
import io
import json
import os
from glob import glob

import cv2
import numpy as np
import paddle
import yaml

import paddle.nn.functional as F
from paddleocr import PaddleOCR
from ppocr.modeling.architectures import build_model
from ppocr.utils.save_load import load_model
from ppocr.data import create_operators, transform
from time import sleep as wait


def read_class_list(filepath):
    dict = {}
    with open(filepath, "r") as f:
        lines = f.readlines()
        for line in lines:
            key, value = line.split(" ")
            dict[key] = value.rstrip()
    return dict


def load_config(file_path):
    """
    Load config from yml/yaml file.
    Args:
        file_path (str): Path of the config file to be loaded.
    Returns: global config
    """
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    config = yaml.load(open(file_path, 'rb'), Loader=yaml.Loader)
    return config


def merge_config(config, opts):
    """
    Merge config into global config.
    Args:
        config (dict): Config to be merged.
    Returns: global config
    """
    for key, value in opts.items():
        if "." not in key:
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
        else:
            sub_keys = key.split('.')
            assert (
                sub_keys[0] in config
            ), "the sub_keys can only be one of global_config: {}, but get: " \
               "{}, please check your running command".format(
                config.keys(), sub_keys[0])
            cur = config[sub_keys[0]]
            for idx, sub_key in enumerate(sub_keys[1:]):
                if idx == len(sub_keys) - 2:
                    cur[sub_key] = value
                else:
                    cur = cur[sub_key]
    return config


def draw_kie_result(batch, node, idx_to_cls, config):
    img = batch[6].copy()
    boxes = batch[7]
    h, w = img.shape[:2]
    pred_img = np.ones((h, w * 2, 3), dtype=np.uint8) * 255
    max_value, max_idx = paddle.max(node, -1), paddle.argmax(node, -1)
    node_pred_label = max_idx.numpy().tolist()
    node_pred_score = max_value.numpy().tolist()

    for i, box in enumerate(boxes):
        if i >= len(node_pred_label):
            break
        new_box = [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]],
                   [box[0], box[3]]]
        Pts = np.array([new_box], np.int32)
        cv2.polylines(
            img, [Pts.reshape((-1, 1, 2))],
            True,
            color=(255, 255, 0),
            thickness=1)
        x_min = int(min([point[0] for point in new_box]))
        y_min = int(min([point[1] for point in new_box]))

        pred_label = str(node_pred_label[i])
        if pred_label in idx_to_cls:
            pred_label = idx_to_cls[pred_label]
        pred_score = '{:.2f}'.format(node_pred_score[i])
        text = pred_label + '(' + pred_score + ')'
        cv2.putText(pred_img, text, (x_min * 2, y_min),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    vis_img = np.ones((h, w * 3, 3), dtype=np.uint8) * 255
    vis_img[:, :w] = img
    vis_img[:, w:] = pred_img
    save_kie_path = os.path.dirname(config['Global'][
        'save_res_path']) + "/kie_results/"
    if not os.path.exists(save_kie_path):
        os.makedirs(save_kie_path)
    save_path = os.path.join(save_kie_path + ".png")
    cv2.imwrite(save_path, vis_img)


def write_kie_result(node, data):
    """
    Write infer result to output file, sorted by the predict label of each line.
    The format keeps the same as the input with additional score attribute.
    """
    import json
    label = data['label']
    annotations = json.loads(label)
    max_value, max_idx = paddle.max(node, -1), paddle.argmax(node, -1)
    node_pred_label = max_idx.numpy().tolist()
    node_pred_score = max_value.numpy().tolist()
    res = []
    for i, label in enumerate(node_pred_label):
        pred_score = '{:.2f}'.format(node_pred_score[i])
        pred_res = {
                'label': label,
                'transcription': annotations[i]['transcription'],
                'score': pred_score,
                'points': annotations[i]['points'],
            }
        res.append(pred_res)
    res.sort(key=lambda x: x['label'])
    return res


def image_processing(imagedata):
    imgdata = base64.b64decode(imagedata)
    stream_img = io.BytesIO(imgdata)
    images = cv2.imdecode(np.frombuffer(stream_img.read(), np.uint8))
    return cv2.cvtColor(images, cv2.COLOR_BGR2RGB)


def get_ocr_result(image):
    ocr = PaddleOCR(use_pdserving=False,
                    use_angle_cls=True,
                    det=True,
                    cls=True,
                    use_gpu=False,
                    lang="en",
                    show_log=False)
    results = ocr.ocr(image, cls=True, det=True)
    end_result = []
    for line in results:
        new_item = {
            "transcription": line[1][0],
            "points": line[0],
            "labels": 0,
            "difficult": False,
            "key_cls": 0,
        }
        end_result.append(new_item)
    return end_result


def combine_result(results):
    end_result = {}
    for index, result in enumerate(results):
        value_label = result["label"]
        score = round(float(result["score"]))
        if value_label == "PROVIDER_NAME" and score == 1:
            end_result["PROVIDER_NAME"] = result["PROVIDER_NAME"]
        if value_label.split("_")[-1] == "value":
            key_label = key_label.replace("value", "key")
            key_exist = 0
            for i in results:
                if i["label"] == key_label and round(float(i["score"])) == 1:
                    end_result[i["label"]] = result[value_label]
                    key_exist = 1
            if key_exist == 0:
                end_result[f"info_{index}"] = value_label
    return end_result


def scan_image(img, label):
    config = load_config("server/shared/helpers/kie/config.yaml")
    class_path = "server/shared/helpers/kie/class_list.txt"
    global_config = config['Global']

    # build model
    model = build_model(config['Architecture'])
    load_model(config, model)

    # create data ops
    transforms = []
    for op in config['Eval']['dataset']['transforms']:
        transforms.append(op)
    ops = create_operators(transforms, global_config)
    idx_to_cls = read_class_list(class_path)
    model.eval()

    data = {'img_path': '.', 'label': label}
    data['image'] = img
    batch = transform(data, ops)
    batch_pred = [0] * len(batch)
    for i in range(len(batch)):
        batch_pred[i] = paddle.to_tensor(
            np.expand_dims(
                batch[i], axis=0))

    node, edge = model(batch_pred)
    node = F.softmax(node, -1)
    draw_kie_result(batch, node, idx_to_cls, config)
    results = write_kie_result(node, data)
    for result in results:
        label = str(result["label"])
        if label in idx_to_cls:
            label = idx_to_cls[label]
            result["label"] = label
    return combine_result(results)


if __name__ == "__main__":
    save_res_path = "server/shared/helpers/result"
    if not os.path.exists(os.path.dirname(save_res_path)):
        os.makedirs(os.path.dirname(save_res_path))
    with open(save_res_path, "w") as fout:
        results = {}
        for file in glob("server/test/id_test/tests/*.jpg"):
            image = open(file, "rb").read()
            file_bytes = np.asarray(bytearray(image), dtype=np.uint8)
            image_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            label = json.dumps(get_ocr_result(image_cv))
            result = scan_image(image, label)
            results[file] = result
            print(result)
        json.dump(results, fout, indent=4)

