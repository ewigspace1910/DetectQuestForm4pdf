#Thirst Library
from ultralytics import YOLO
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
        
        #matpix
        self.matpix_id = os.getenv("mathpix_id") 
        self.matpix_key = os.getenv("mathpix_key") 

        #storage
        # self.s3_id = os.getenv("s3id")
        # self.s3_key = os.getenv("s3key")
        self.s3_bucket_name = os.getenv("s3buckname")

        #parallel
        self.n_processes = os.getenv("nprocess") #for aws
        #detection model
        self.store_path =  os.getenv("wstore_path") #"deploy/model/yolov8x.pt"


################################################
#               RESQUEST
#################################################
def infer_yolov8(source, model):
    """
    source:  could be image, path,... 
    """
    detection_results = model.predict(source, save=False, imgsz=(640, 640), conf=0.5, device="cpu")
    return detection_results

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
    detector = Detector(configs=Config())
    return detector

# @profile
def subprocess_ocr_request_aws(page_index, box, cropped_image, mathpix_id, mathpix_key, s3name, conn):
    """
    "inputs" contains:
        page_index  : index of page containing box
        box         : detectedbox list from yolov8 results
        cropped_image : (numpy array): original imaege
        app_id, app_key
    """
    cls = REVERSE_MAP[int(box.cls)]
    _, buffer = cv2.imencode(".png", cropped_image)
    io_buffer = io.BytesIO(buffer)
    io_buffer.seek(0)
    if cls in ["image", "table"]: 
        text = request2s3(file=io_buffer, s3name=s3name)
    else: 
        text = request2mathpix(io_buffer=io_buffer, app_id=mathpix_id, app_key=mathpix_key)

    obj = { "name":cls, 
            "title": text,
            "stimulus": "", #save image/table
            "prompt" : "", #auxilary_text
            "category": "OEQ",
            "idx": (page_index, int(box.xyxy[0][1]))}
    
    conn.send(obj)
    conn.close()


class Detector(object):
    def __init__(self, configs:Config):
        self.cfg = configs
        self.model = YOLO(self.cfg.store_path)
        self.infer_func = infer_yolov8
    
    def ready(self):
        return str(type(self.model))
   
    # @profile
    def infer_concurrent(self, inputs):
        """
        inputs: upload file from http requestion

        Return --> list(dict) : list of question objects with their child elements
        """
        #Prediction
        detection_results = self.infer_func(source=inputs, model=self.model)

        #OCR                      
        parent_connections = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for page_index, results in enumerate(detection_results):
                for box in results.boxes:
                    parent_conn, child_conn = mp.Pipe()
                    parent_connections += [parent_conn]
                            
                    executor.submit(subprocess_ocr_request_aws, page_index, box,
                                detection_results[page_index].orig_img[int(box.xyxy[0][1]): int(box.xyxy[0][3]), int(box.xyxy[0][0]): int(box.xyxy[0][2])],
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
            tmp = []
            tmp_e = []
            new_arr = []
            new_arr_e =[]
            i = len(arr)-1
            while i >= 0:
                if arr[i] > order:
                    new_arr = [arr[i], tmp] + new_arr if len(tmp) > 0 else [arr[i]] + new_arr
                    new_arr_e = [idx_e[i], tmp_e[::-1]] + new_arr_e if len(tmp_e) > 0 else [idx_e[i]] + new_arr_e
                    tmp = []
                    tmp_e = []
                else:
                    tmp = [arr[i]]
                    tmp_e += [idx_e[i]]
                i -= 1
            #reduce array for [2, [0, 0], 3 [0], 1 [0, 0] 1 [0] 3 2 [0] 1 [0] ] -> [2, 3, 1, 1, 3, 2, 1]
            new_arr_2 = []
            new_arr_2_e = []
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
                        # elements[new_arr_e[i-1]]["child"] += [ elements[j] ] 
                i += 1

            new_arr_2_e = [elements[i] for i in new_arr_2_e]

            return new_arr_2, new_arr_2_e
        
        #######################
        # orders = [CATEGORY_LEVEL[obj['name']] for obj in elements]
        ## exclude unessesary headings --> example: 332211031032230212102 --> 3221101022302121102
        new_elements=[]
        orders   =[]
        for i, item in enumerate(elements):
            order = CATEGORY_LEVEL[item['name']]
            if order == 3 and i > 0:
                if CATEGORY_LEVEL[elements[i-1]['name']] == 3:
                    elements[i-1]['prompt'] += f"\n{item['title'].strip()}"
                    continue
                elif i== len(elements) or CATEGORY_LEVEL[elements[i+1]['name']] != 0: continue
            
            del item["idx"]
            orders += [order]
            new_elements += [item]
        
        ## Grouping items
        if orders[0] < 3:
            orders = [3] + orders
            elements = [{"name":'heading', 
                        "title": "Section - 1",
                        "stimulus": "", #save image/table
                        "prompt" : "", #auxilary_text
                        "category": "MSQ"}] + elements
        o, e = orders, elements
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

    for item in items:
        match = re.match(r"\((\w+)\)\s*(\S+)", item)

        if match:
            value = match.group(2)
            results += [value]
    if len(results) == 0 : return [text]
    else: return results