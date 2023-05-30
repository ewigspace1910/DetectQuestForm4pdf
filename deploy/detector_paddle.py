#Thirst Library
import multiprocessing as mp
import json
import cv2
import io
import requests
import boto3
import os
from botocore import UNSIGNED
from botocore.client import Config as AWSConfig
import time
import random
import concurrent.futures
import re
import yaml
import numpy as np
import math
from paddle.inference import Config as PPConfig
from paddle.inference import create_predictor

# CONSTANTs
CATEGORY_MAP = {
    "heading": 0, 
    "question": 1,  
    "subquestion": 2,
    "choice"  : 3,
    "image"   : 4,
    "table"   : 5,
    "blank"   : 6,
    "auxillary_text"   : 7
}
CATEGORY_LEVEL = {
    "heading": 3, 
    "question": 2,  
    "subquestion": 1,
    "choice"  : 0,
    "image"   : 0,
    "table"   : 0,
    "blank"   : 0,
    "auxillary_text"   : 0
}
REVERSE_MAP = {CATEGORY_MAP[k]:k for k in CATEGORY_MAP}
class Config:
    def __init__(self):
        
        #key-id
        self.matpix_id = os.getenv("mathpix_id") 
        self.matpix_key = os.getenv("mathpix_key") 
        self.s3_bucket_name = os.getenv("s3buckname")

        #detection model
        self.store_path =  "data/model/rt-dert" #os.getenv("wstore_path") 
        self.cfg_path   =   "deploy/config.yaml" #os.getenv("cfg_path") 
        #same args
        self.threshold  = 0.5 if os.getenv("threshold") is None else os.getenv("threshold")
        self.device     = "cpu" if os.getenv("device") is None else os.getenv("device")
        self.n_processes = 2 #os.getenv("nprocess") #for aws

################################################
#               RESQUEST
#################################################
def infer_rtdert(source, model):
    return model.predict_image(source)

################## REQUEST #####################
def request2mathpix(io_buffer, app_id="", app_key=""):
    try:
        r = requests.post("https://api.mathpix.com/v3/text",
            files={"file": io_buffer},
            data={
            "options_json": json.dumps({
                "math_inline_delimiters": ["$", "$"],
                "rm_spaces": True
            })
            },
            headers={
                "app_id": app_id ,
                "app_key":app_key 
            }
        )
        r = r.json()
        if "text" in r.keys():
            return r['text'].strip()
        else :
            return ""
    except Exception as e:
        print("Mathpix_ocr_requestion got error :", e)
        return "???"

def request2s3(file, s3name):
        file_name = "image/" + time.strftime("%Y%m%d-%H%M%S") + "-" + str(random.randint(1000, 9999)) + ".png"
        try:
            s3 = boto3.client('s3', config=AWSConfig(signature_version=UNSIGNED))
            s3.upload_fileobj(file, s3name, file_name) #upload to s3.
            text = "https://{}.s3.amazonaws.com/{}".format(s3name, file_name)
            return text
        except Exception as e:
            print("S3 error", e)
            return ""   
#########################################
### 
##       PIPELINE
###
#########################################
def get_detector():
    detector = DetectorSJ(configs=Config())
    return detector

# @profile
def subprocess_ocr_request(idx, cls, cropped_image, mathpix_id, mathpix_key, s3name, conn):
    """
    "inputs" contains:
        box         : detectedbox list from paddle results
        cropped_image : (numpy array): original imaege
        app_id, app_key
    """
    cls = REVERSE_MAP[cls]
    _, buffer = cv2.imencode(".png", cropped_image)
    io_buffer = io.BytesIO(buffer)
    io_buffer.seek(0)
    if cls in ["image", "table"]: 
        text = request2s3(file=io_buffer, s3name=s3name)
    else: 
        text = request2mathpix(io_buffer=io_buffer, app_id=mathpix_id, app_key=mathpix_key)

    obj = { "name":cls, 
            "title": str(idx),#text,
            "stimulus": "", #save image/table
            "prompt" : "", #auxilary_text
            "category": "OEQ",
            "idx": idx}
    
    conn.send(obj)
    conn.close()


class DetectorSJ(object):
    def __init__(self, configs:Config):
        self.cfg = configs
        self.infer_func = infer_rtdert 
        if not os.path.isdir(self.cfg.store_path): raise f"{self.cfg.store_path} has to be folder contain  model.pdiparams, model.pdmodel"  
        self.model = PaddleDectector(
            self.cfg.store_path,
            self.cfg.cfg_path,
            device=self.cfg.device,
            cpu_threads=self.cfg.n_processes,
            threshold=self.cfg.threshold)

    # predict from image
    def ready(self):
        return str(type(self.model))
   
    # @profile
    def infer_concurrent(self, inputs):
        """
        inputs: upload file from http requestion

        Return --> list(dict) : list of question objects with their child elements
        """
        #Prediction
        inputs = [np.array(i) for i in inputs]
        detection_results = self.infer_func(source=inputs, model=self.model)

        #OCR            
        parent_connections = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for box in detection_results:
                parent_conn, child_conn = mp.Pipe()
                parent_connections += [parent_conn]      
                executor.submit(subprocess_ocr_request, (box['page_id'], box['xyxy'][1]), box['cls'],
                            inputs[box['page_id']][box['xyxy'][1]: box['xyxy'][3], box['xyxy'][0]: box['xyxy'][2]],
                            self.cfg.matpix_id, self.cfg.matpix_key, self.cfg.s3_bucket_name,
                            child_conn)
        elements = []
        for parent_connection in parent_connections:
            elements += [parent_connection.recv()]

        #Postprocesing
        elements = sorted(elements, key=lambda x: x["idx"])
        questions = self.__group_element(elements)

        return questions     

    def __group_element(self, elements):
        """Args:
            - orders : (list[int])   Contain category_weights corresponding elements
            - elements: (list[dict]) Contain all detected element in order from top to bottom of page, from page 0 to page n
            Return: json object like html document object
        """
        def __combine_min_elements(orders, elements):
            '''
            order (list) --> rank of elements
            elements (list) --> containing dictionary objects  contain information of question, must be containt key["child"] = []
            '''
            arr = orders
            idx_e = [i for i in range(len(elements))]
            order = min(orders)
            #Group min elements in orders list : [2 0 0 3 0 1 0 0 1 0 3 2 0 1 0  ] -> [2, [0, 0], 3 [0], 1 [0, 0] 1 [0] 3 2 [0] 1 [0] ]
            tmp, tmp_e, new_arr, new_arr_e =[], [], [], []
            i = len(arr)-1
            while i >= 0:
                if arr[i] > order:
                    new_arr = [arr[i], tmp] + new_arr if len(tmp) > 0 else [arr[i]] + new_arr
                    new_arr_e = [idx_e[i], tmp_e[::-1]] + new_arr_e if len(tmp_e) > 0 else [idx_e[i]] + new_arr_e
                    tmp, tmp_e = [], []
                else:
                    tmp += [arr[i]]
                    tmp_e += [idx_e[i]]
                i -= 1
            #reduce array for [2, [0, 0], 3 [0], 1 [0, 0] 1 [0] 3 2 [0] 1 [0] ] -> [2, 3, 1, 1, 3, 2, 1]
            new_arr_2, new_arr_2_e = [], []
            i = 0
            while i < len(new_arr):
                if type(new_arr[i]) == int:
                    new_arr_2 += [new_arr[i]]
                    new_arr_2_e += [new_arr_e[i]]
                else:
                    for j in new_arr_e[i]:
                        if elements[j]["name"] in ("image", "table"):
                            elements[new_arr_e[i-1]]["stimulus"] += f"\n{elements[j]['title']}" 
                        elif elements[j]["name"] in ("auxillary_text") and len(elements[j]["title"]) > 5:
                            elements[new_arr_e[i-1]]["prompt"] += f"\n{elements[j]['title']}"

                        elif elements[j]["name"] in ("blank") and len(elements[j]['title']) > 10:
                            elements[new_arr_e[i-1]]["prompt"] += f"\n{elements[j]['title']}"
                            # elements[new_arr_e[i-1]]["category"] = "OEQ"

                        elif elements[j]["name"] in ("choice"):
                            if "choices" in elements[new_arr_e[i-1]].keys():
                                elements[new_arr_e[i-1]]["choices"] += parse_choice2dict(elements[j]['title'])#f"\n{elements[j]['title']}"
                            else: elements[new_arr_e[i-1]]["choices"] = parse_choice2dict(elements[j]['title']) #f"\n{elements[j]['title']}"
                            elements[new_arr_e[i-1]]["category"] = "MCQ"

                        elif elements[j]["name"] in ("subquestion", "question"):
                            if "subquestions" in elements[new_arr_e[i-1]].keys():
                                elements[new_arr_e[i-1]]["subquestions"] += [ elements[j] ]
                            else: elements[new_arr_e[i-1]]["subquestions"] = [ elements[j] ]
                            elements[new_arr_e[i-1]]["category"] = "MSQ"
                        else: pass 
                i += 1

            new_arr_2_e = [elements[i] for i in new_arr_2_e]
            return new_arr_2, new_arr_2_e
        
        #######################
        ## exclude unessesary headings --> example: 3322110310322302121023 --> 3221101022302121102
        new_elements=[]
        orders   =[]
        for i, item in enumerate(elements):
            order = CATEGORY_LEVEL[item['name']]
            if order == 3 and i > 0:
                if CATEGORY_LEVEL[elements[i-1]['name']] == 3:
                    elements[i-1]['prompt'] += f"\n{item['title'].strip()}"
                    continue
                elif i== len(elements)-1:continue
                elif CATEGORY_LEVEL[elements[i+1]['name']] in [1, 2]: continue
            
            del item["idx"]
            orders += [order]
            new_elements += [item]
        ## Grouping items
        if orders[0] < 3:
            orders = [3] + orders
            new_elements = [{"name":'heading', 
                        "title": "Section - 1",
                        "stimulus": "", #save image/table
                        "prompt" : "", #auxilary_text
                        "category": "MSQ"}] + new_elements
        o, e = orders, new_elements
        while len(set(o)) > 1:
            o, e = __combine_min_elements(o,e)
        return e

#########################################
### 
##       POSTPROCESSING
###
#########################################
def parse_choice2dict(text):
    items = [line.strip() for line in text.splitlines() if line.strip()] 
    results = []
    return items
    for item in items:
        match = re.match(r"\((\w+)\)\s*(\S+)", item)

        if match:
            value = match.group(2)
            results += [value]
    if len(results) == 0 : return [text]
    else: return results


#########################################
### 
##       Paddle module
###
#########################################
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Source : https://github.com/PaddlePaddle/PaddleDetection/tree/develop/deploy/python
"""

def decode_image(im_file, im_info):
    """read rgb image
    Args:
        im_file (str|np.ndarray): input can be image path or np.ndarray
        im_info (dict): info of image
    Returns:
        im (np.ndarray):  processed image (np.ndarray)
        im_info (dict): info of processed image
    """
    if isinstance(im_file, str):
        with open(im_file, 'rb') as f:
            im_read = f.read()
        data = np.frombuffer(im_read, dtype='uint8')
        im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    else:
        im = im_file
    im_info['im_shape'] = np.array(im.shape[:2], dtype=np.float32)
    im_info['scale_factor'] = np.array([1., 1.], dtype=np.float32)
    return im, im_info


class Resize(object):
    """resize image by target_size and max_size
    Args:
        target_size (int): the target size of image
        keep_ratio (bool): whether keep_ratio or not, default true
        interp (int): method of resize
    """

    def __init__(self, target_size, keep_ratio=True, interp=cv2.INTER_LINEAR):
        if isinstance(target_size, int):
            target_size = [target_size, target_size]
        self.target_size = target_size
        self.keep_ratio = keep_ratio
        self.interp = interp

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        assert len(self.target_size) == 2
        assert self.target_size[0] > 0 and self.target_size[1] > 0
        im_channel = im.shape[2]
        im_scale_y, im_scale_x = self.generate_scale(im)
        im = cv2.resize(
            im,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=self.interp)
        im_info['im_shape'] = np.array(im.shape[:2]).astype('float32')
        im_info['scale_factor'] = np.array(
            [im_scale_y, im_scale_x]).astype('float32')
        return im, im_info

    def generate_scale(self, im):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
        Returns:
            im_scale_x: the resize ratio of X
            im_scale_y: the resize ratio of Y
        """
        origin_shape = im.shape[:2]
        im_c = im.shape[2]
        if self.keep_ratio:
            im_size_min = np.min(origin_shape)
            im_size_max = np.max(origin_shape)
            target_size_min = np.min(self.target_size)
            target_size_max = np.max(self.target_size)
            im_scale = float(target_size_min) / float(im_size_min)
            if np.round(im_scale * im_size_max) > target_size_max:
                im_scale = float(target_size_max) / float(im_size_max)
            im_scale_x = im_scale
            im_scale_y = im_scale
        else:
            resize_h, resize_w = self.target_size
            im_scale_y = resize_h / float(origin_shape[0])
            im_scale_x = resize_w / float(origin_shape[1])
        return im_scale_y, im_scale_x


class NormalizeImage(object):
    """normalize image
    Args:
        mean (list): im - mean
        std (list): im / std
        is_scale (bool): whether need im / 255
        norm_type (str): type in ['mean_std', 'none']
    """

    def __init__(self, mean, std, is_scale=True, norm_type='mean_std'):
        self.mean = mean
        self.std = std
        self.is_scale = is_scale
        self.norm_type = norm_type

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        im = im.astype(np.float32, copy=False)
        if self.is_scale:
            scale = 1.0 / 255.0
            im *= scale

        if self.norm_type == 'mean_std':
            mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
            std = np.array(self.std)[np.newaxis, np.newaxis, :]
            im -= mean
            im /= std
        return im, im_info


class Permute(object):
    """permute image
    Args:
        to_bgr (bool): whether convert RGB to BGR 
        channel_first (bool): whether convert HWC to CHW
    """

    def __init__(self, ):
        super(Permute, self).__init__()

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        im = im.transpose((2, 0, 1)).copy()
        return im, im_info




def preprocess(im, preprocess_ops):
    # process image by preprocess_ops
    im_info = {
        'scale_factor': np.array(
            [1., 1.], dtype=np.float32),
        'im_shape': None,
    }
    im, im_info = decode_image(im, im_info)
    for operator in preprocess_ops:
        im, im_info = operator(im, im_info)
    return im, im_info

###########################################################################################################3

# DETECTOR


##########################################################################################################
       
class PaddleDectector(object):
    #https://github.com/PaddlePaddle/PaddleDetection/tree/develop/deploy/python/infer.py

    def __init__(self,
                 model_dir,
                 cfg_path,
                 device='CPU',
                 cpu_threads=2,
                 threshold=0.5,):
        self.pred_config = PredictConfig(cfg_path)
        self.predictor, self.config = load_predictor( model_dir,run_mode=self.pred_config.mode, device=device, cpu_threads=cpu_threads)
        self.batch_size = self.pred_config.batch_size
        self.threshold = threshold

    def preprocess(self, image_list):
        preprocess_ops = []
        for op_info in self.pred_config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            preprocess_ops.append(eval(op_type)(**new_op_info))
        input_im_lst = []
        input_im_info_lst = []
        for pil_img in image_list:
            pil_img = np.array(pil_img)
            im, im_info = preprocess(pil_img, preprocess_ops)
            input_im_lst.append(im)
            input_im_info_lst.append(im_info)

        inputs = create_inputs(input_im_lst, input_im_info_lst)
        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            if input_names[i] == 'x':
                input_tensor.copy_from_cpu(inputs['image'])
            else:
                input_tensor.copy_from_cpu(inputs[input_names[i]])

        return inputs

    # def postprocess(self, inputs, result):
    #     # postprocess output of predictor
    #     np_boxes_num = result['boxes_num']
    #     assert isinstance(np_boxes_num, np.ndarray), '`np_boxes_num` should be a `numpy.ndarray`'

    #     result = {k: v for k, v in result.items() if v is not None}
    #     return result

    def predict(self):
        '''
        Args:
            repeats (int): repeats number for prediction
        Returns:
            result (dict): include 'boxes': np.ndarray: shape:[N,6], N: number of box,
                            matix element:[class, score, x_min, y_min, x_max, y_max]
                            MaskRCNN's result include 'masks': np.ndarray:
                            shape: [N, im_h, im_w]
        '''
        # model prediction
        np_boxes_num, np_boxes = np.array([0]), None

        self.predictor.run()
        output_names = self.predictor.get_output_names()
        boxes_tensor = self.predictor.get_output_handle(output_names[0])
        np_boxes = boxes_tensor.copy_to_cpu()
        if len(output_names) == 1:
            # some exported model can not get tensor 'bbox_num' 
            np_boxes_num = np.array([len(np_boxes)])
        else:
            boxes_num = self.predictor.get_output_handle(output_names[1])
            np_boxes_num = boxes_num.copy_to_cpu()
        result = dict(boxes=np_boxes, boxes_num=np_boxes_num)
        return result

    def merge_batch_result(self, batch_result):
        if len(batch_result) == 1:
            return batch_result[0]
        res_key = batch_result[0].keys()
        results = {k: [] for k in res_key}
        for res in batch_result:
            for k, v in res.items():
                results[k].append(v)
        for k, v in results.items():
            if k not in ['masks', 'segm']:
                results[k] = np.concatenate(v)
        return results


    def predict_image(self, image_list):
        batch_loop_cnt = math.ceil(float(len(image_list)) / self.batch_size)
        results = []
        for i in range(batch_loop_cnt):
            start_index = i * self.batch_size
            end_index = min((i + 1) * self.batch_size, len(image_list))
            batch_image_list = image_list[start_index:end_index]
            # preprocess
            inputs = self.preprocess(batch_image_list)
            # model prediction
            result = self.predict()
            # postprocess
            result = {k: v for k, v in result.items() if v is not None} #self.postprocess(inputs, result)
            results.append(result)
        results = self.merge_batch_result(results)

        bbox_results = []
        idx = 0
        for i, box_num in enumerate(results['boxes_num']):
            img_id = i
            if 'boxes' in results:
                boxes = results['boxes'][idx:idx + box_num] 
                boxes = boxes[ boxes[:,1] > self.threshold].tolist()
                bbox_results += [{
                    'page_id': img_id,
                    'cls': int(box[0]),
                    'xyxy': [int(box[2]), int(box[3]), int(box[4]+1), int(box[5]+1)]  # xyxy 
                    } for box in boxes]
            idx += box_num
        return bbox_results


def create_inputs(imgs, im_info):
    """generate input for different model type
    Args:
        imgs (list(numpy)): list of images (np.ndarray)
        im_info (list(dict)): list of image info
    Returns:
        inputs (dict): input of model
    """
    inputs = {}

    im_shape = []
    scale_factor = []
    if len(imgs) == 1:
        inputs['image'] = np.array((imgs[0], )).astype('float32')
        inputs['im_shape'] = np.array(
            (im_info[0]['im_shape'], )).astype('float32')
        inputs['scale_factor'] = np.array(
            (im_info[0]['scale_factor'], )).astype('float32')
        return inputs

    for e in im_info:
        im_shape.append(np.array((e['im_shape'], )).astype('float32'))
        scale_factor.append(np.array((e['scale_factor'], )).astype('float32'))

    inputs['im_shape'] = np.concatenate(im_shape, axis=0)
    inputs['scale_factor'] = np.concatenate(scale_factor, axis=0)

    imgs_shape = [[e.shape[1], e.shape[2]] for e in imgs]
    max_shape_h = max([e[0] for e in imgs_shape])
    max_shape_w = max([e[1] for e in imgs_shape])
    padding_imgs = []
    for img in imgs:
        im_c, im_h, im_w = img.shape[:]
        padding_im = np.zeros(
            (im_c, max_shape_h, max_shape_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = img
        padding_imgs.append(padding_im)
    inputs['image'] = np.stack(padding_imgs, axis=0)
    return inputs


class PredictConfig():
    """set config of preprocess, postprocess and visualize
    Args:
        model_dir (str): root path of model.yml
    """

    def __init__(self, cfg_path):
        # parsing Yaml config for Preprocess
        with open(cfg_path) as f:
            yml_conf = yaml.safe_load(f)
        self.arch = yml_conf['arch']
        self.mode = yml_conf['mode']
        self.preprocess_infos = yml_conf['Preprocess']
        self.labels = yml_conf['label_list']
        self.batch_size = yml_conf["bs"]


def load_predictor(model_dir,
                   run_mode='paddle',
                   device='CPU',
                   cpu_threads=2, **kwargs):
    if device != 'GPU' and run_mode != 'paddle':
        raise ValueError("Predict by TensorRT mode: {}, expect device=='GPU', but device == {}".format(run_mode, device))
    infer_model = os.path.join(model_dir, 'model.pdmodel')
    infer_params = os.path.join(model_dir, 'model.pdiparams')
    config = PPConfig(infer_model, infer_params)
    if device == 'GPU':
        config.enable_use_gpu(200, 0)         # initial GPU memory(M), device ID
        config.switch_ir_optim(True)         # optimize graph and fuse op
    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(cpu_threads)

    # disable print log when predict
    config.disable_glog_info()
    # enable shared memory
    config.enable_memory_optim()
    # disable feed, fetch OP, needed by zero_copy_run
    config.switch_use_feed_fetch_ops(False)

    predictor = create_predictor(config)
    return predictor, config