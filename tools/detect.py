import os
import argparse
import time
import _thread

import _init_paths  # pylint: disable=unused-import

from urllib import request
import cv2
import numpy as np
import base64
import urllib3
import uuid

import torch

from rcnn.core.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg
from rcnn.modeling.parsing_rcnn.inference import parsing_results
from rcnn.core.test_engine import initialize_model_from_cfg
import rcnn.core.test as rcnn_test

# Parse arguments
parser = argparse.ArgumentParser(description='Hier R-CNN Detect')
parser.add_argument('--cfg', dest='cfg_file',
                    help='optional config file',
                    default='./cfgs/mscoco_humanparts/e2e_hier_rcnn_R-50-FPN_1x.yaml', type=str)
parser.add_argument('--gpu_id', type=str, default='0,1,2,3,4,5,6,7', help='gpu id for evaluation')
parser.add_argument('opts', help='See rcnn/core/config.py for all options',
                    default=None,
                    nargs=argparse.REMAINDER)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

# http连接池
http_client = urllib3.PoolManager()

def main():
    if len(args.gpu_id.split(',')) == 1:
        local_rank = int(args.gpu_id.split(',')[0])
    else:
        local_rank = -1
    args.local_rank = local_rank

    num_gpus = len(args.gpu_id.split(','))
    multi_gpu_testing = True if num_gpus > 1 else False

    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.opts is not None:
        merge_cfg_from_list(args.opts)

    assert_and_infer_cfg(make_immutable=False)
    args.test_net_file, _ = os.path.splitext(__file__)

    model = initialize_model_from_cfg()

    start_time = time.time()
    image = cv2.imread('', cv2.IMREAD_COLOR)
    box_results, par_results, par_score = detect(model, image)
    print(' cost: ' + str(time.time() - start_time))

    print(dict(
        boxes=box_results,
        parss=par_results,
        pscores=par_score,
    ))

def get_image_from_base64(base64_code):
    ''' 
    base64转成opencv的图片对象
    '''
    img_data = base64.b64decode(base64_code)
    return read_image(img_data)

def get_image_from_url(url):
    '''
    把图片url转成opencv的图片对象
    '''
    img_data = http_client.request("GET", url).data
    print(img_data.count)
    return read_image(img_data)

def read_image(img_data):
    '''
    把字节数组转成opencv的图片对象
    '''
    imgArray = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(imgArray, cv2.IMREAD_COLOR)
    # cv2.imwrite('/Users/kevin/Downloads/2020.jpg', img)
    return img


def detect(model, image):
    start_time = time.time()
    with torch.no_grad():
        results, features = rcnn_test.im_detect_bbox(model, [image])
        print("1 cost: " + str(time.time() - start_time))
        start_time = time.time()

        if cfg.MODEL.MASK_ON:
            result = rcnn_test.im_detect_mask(model, results, features)
            print("2 cost: " + str(time.time() - start_time))
            start_time = time.time()
        if cfg.MODEL.PARSING_ON:
            result = rcnn_test.im_detect_parsing(model, results, features)
            print("3 cost: " + str(time.time() - start_time))
            start_time = time.time()

        if not results or len(results) != 1 or len(results[0]) == 0:
            return None

        image_height = image.shape[0]
        image_width = image.shape[1]

        cpu_device = torch.device("cpu")
        result = result[0].to(cpu_device)
        result = result.resize((image_width, image_height))
        
        return post_processing(result, image)

def post_processing(result, image):
    start_time = time.time()
    box_results = prepare_box_results(result, image)
    print("4 cost: " + str(time.time() - start_time))
    start_time = time.time()

    if cfg.MODEL.PARSING_ON:
        par_results, par_score = prepare_parsing_results(result, image)
        print("5 cost: " + str(time.time() - start_time))
        start_time = time.time()
    else:
        par_results = []
        par_score = []

    return box_results, par_results, par_score

def prepare_box_results(result, image):
    scores = result.get_field("scores").tolist()
    result = result.convert("xywh")
    boxes = result.bbox.tolist()

    return [
            {
                "bbox": box,
                "score": scores[k],
            }
            for k, box in enumerate(boxes)
        ]

def prepare_parsing_results(result, image):
    semseg = result.get_field("semseg") if cfg.MODEL.SEMSEG_ON else None
    parsing = result.get_field("parsing")
    parsing = parsing_results(parsing, result, semseg=semseg)
    scores = result.get_field("parsing_scores")

    return parsing, scores

if __name__ == '__main__':
    main()