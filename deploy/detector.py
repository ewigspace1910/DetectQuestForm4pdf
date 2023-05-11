#Thirst Library
from ultralytics import YOLO
import multiprocessing as mp
import json
import cv2
import io
import requests
import boto3
import cloudinary
import cloudinary.uploader as CryUploader
from botocore.exceptions import NoCredentialsError
import time
import random
import concurrent.futures

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
        self.matpix_id = ""
        self.matpix_key = ""


        #storage
        self.s3_id = ""
        self.s3_key = ""
        self.s3_bucket_name = ""

        self.cloudinary_name = ""
        self.cloudinary_key = ""
        self.cloudinary_api_secret = "dhvcDtRatChtiNr021XALeBPWU0"


        #detection model
        self.model_type = "yolov8"
        self.weight_path = "deploy/model/yolov8.pt"

        #parallel
        self.n_processes = 2 #for aws

################################################
#               RESQUEST
#################################################
def infer_yolov8(source, model):
    """
    source:  could be image, path,... 
    """
    detection_results = []
    detection_results = model.predict(source, save=False, imgsz=(640, 640), conf=0.45, device="cpu")
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
            return r['text']
        else :
            return ""
    except Exception as e:
        print("Mathpix_ocr_requestion got error :", e)
        return "???"

def request2cloudinary(file, name, key, api_secret):
    try:
        cloudinary.config(cloud_name = name, api_key = key, api_secret = api_secret, secure = True)
        r = CryUploader.upload(file, format="png")
        return r['url']
    except Exception as e:
        print("cloudary_error", e)
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
def subprocess_ocr_request(page_index, box, cropped_image, mathpix_id, mathpix_key, cry_name,  cry_key, cry_id):
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
        text = request2cloudinary(file=io_buffer, name=cry_name, key=cry_key, api_secret=cry_id)
    else: 
        text = request2mathpix(io_buffer=io_buffer, app_id=mathpix_id, app_key=mathpix_key)

    obj = { "name":cls, 
            "title": text,
            "stimulus": "", #save image/table
            "prompt" : "", #auxilary_text
            "category": "OEQ",
            "idx": (page_index, int(box.xyxy[0][1]))}
    return obj
  
def subprocess_ocr_request_aws(page_index, box, cropped_image, mathpix_id, mathpix_key, cry_name,  cry_key, cry_id, conn):
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
        text = request2cloudinary(file=io_buffer, name=cry_name, key=cry_key, api_secret=cry_id)
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
        # self.s3 = boto3.client('s3', aws_access_key_id=self.cfg.s3_id, aws_secret_access_key=self.cfg.s3_key)

        if self.cfg.model_type == "yolov8":
            self.model = YOLO(self.cfg.weight_path)
            self.infer_func = infer_yolov8
    
    def ready(self):
        return str(type(self.model))

    # @profile
    def infer(self, inputs): 
        """
        This function takes inputs, performs object detection on them, extracts text from the detected
        objects, and groups the extracted text into questions.
        
        :param inputs: The input data for the inference process. It is passed to the `infer_func` method
        along with the model to obtain detection results
        :return: a list of dictionaries, where each dictionary represents an element detected in the
        input image(s) and contains information such as the element's name (e.g. "image", "table",
        "heading"), text (if applicable), and its position in the image(s). The elements are sorted by
        their position and grouped into categories (e.g. headings, paragraphs, images) based on
        """
        #DETECTION
        detection_results = self.infer_func(source=inputs, model=self.model)

        #OCR
        elements = []
        for page_index, result in enumerate(detection_results):
            for box in result.boxes:
                obj = subprocess_ocr_request(page_index, box,
                            detection_results[page_index].orig_img[int(box.xyxy[0][1]): int(box.xyxy[0][3]), int(box.xyxy[0][0]): int(box.xyxy[0][2])],
                            self.cfg.matpix_id, self.cfg.matpix_key,
                            self.cfg.cloudinary_name, self.cfg.cloudinary_key, self.cfg.cloudinary_api_secret)
                elements += [obj]

        #Postprocessing
        elements = sorted(elements, key=lambda x: x["idx"])
        orders = [CATEGORY_LEVEL[obj['name']] for obj in elements]        
        questions = self.__group_element(orders, elements)
        return questions
        

    def infer_parallel(self, inputs):
        """
        inputs: upload file from http requestion

        Return --> list(dict) : list of question objects with their child elements
        """
        #Prediction
        detection_results = self.infer_func(source=inputs, model=self.model)


        #OCR
        with mp.Pool(processes=self.cfg.n_processes) as executor:
            elements = executor.starmap(subprocess_ocr_request, 
                            [(page_index, box,
                            detection_results[page_index].orig_img[int(box.xyxy[0][1]): int(box.xyxy[0][3]), int(box.xyxy[0][0]): int(box.xyxy[0][2])],
                            self.cfg.matpix_id, self.cfg.matpix_key,
                            self.cfg.cloudinary_name, self.cfg.cloudinary_key, self.cfg.cloudinary_api_secret) for page_index, results in enumerate(detection_results) for box in results.boxes])
                            
        #Postprocesing
        elements = sorted(elements, key=lambda x: x["idx"])
        orders = [CATEGORY_LEVEL[obj['name']] for obj in elements]
        questions = self.__group_element(orders, elements)
        return questions 
    
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
                                self.cfg.matpix_id, self.cfg.matpix_key,
                                self.cfg.cloudinary_name, self.cfg.cloudinary_key, self.cfg.cloudinary_api_secret,
                                child_conn)
        elements = []
        for parent_connection in parent_connections:
            elements += [parent_connection.recv()]

        #Postprocesing
        elements = sorted(elements, key=lambda x: x["idx"])
        orders = [CATEGORY_LEVEL[obj['name']] for obj in elements]
        questions = self.__group_element(orders, elements)

        return questions     


    def __group_element(self, orders, elements):
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
                            elements[new_arr_e[i-1]]["stimulus"] += f"\n{elements[j]['title']}\t" 
                        elif elements[j]["name"] in ("auxillary_text"):
                            elements[new_arr_e[i-1]]["prompt"] += f"\n{elements[j]['title']}"

                        elif elements[j]["name"] in ("blank"):
                            elements[new_arr_e[i-1]]["prompt"] += f"\n{elements[j]['title']}\n"
                            elements[new_arr_e[i-1]]["category"] = "OEQ"

                        elif elements[j]["name"] in ("choice"):
                            if "choices" in elements[new_arr_e[i-1]].keys():
                                elements[new_arr_e[i-1]]["choices"] += f"\n{elements[j]['title']}"
                            else: elements[new_arr_e[i-1]]["choices"] = f"\n{elements[j]['title']}"
                            elements[new_arr_e[i-1]]["category"] = "MCQ"

                        elif elements[j]["name"] in ("subquestion", "question"):
                            if "subquestions" in elements[new_arr_e[i-1]].keys():
                                elements[new_arr_e[i-1]]["subquestions"] += [ elements[j] ]
                            else: elements[new_arr_e[i-1]]["subquestions"] = [ elements[j] ]
                            elements[new_arr_e[i-1]]["category"] = "SQ"
                        else: pass
                        # elements[new_arr_e[i-1]]["child"] += [ elements[j] ] 
                i += 1

            new_arr_2_e = [elements[i] for i in new_arr_2_e]

            return new_arr_2, new_arr_2_e

        #run
        if orders[0] < 3:
            orders = [3] + orders
            elements = [{"name":'heading', 
                        "title": "Section - 1 ",
                        "stimulus": "", #save image/table
                        "prompt" : "", #auxilary_text
                        "category": "SQ", 
                        "idx": (0, 0)}] + elements
        o, e = orders, elements
        while len(set(o)) > 1:
            o, e = __combine_min_elements(o,e)
        return e

    def __upload_file_s3(self, bucket_name, file):
        file_name = time.strftime("%Y%m%d-%H%M%S") + "-" + str(random.randint(1000, 9999)) + ".png"
        try:
            self.s3.upload_fileobj(file, bucket_name, file_name) #upload to s3.
            text = "http://{}.s3.amazonaws.com/{}".format(self.cfg.s3_bucket_name, file_name)
            return text
        except FileNotFoundError:
            print("The file was not found")
            return ""
        except NoCredentialsError:
            print("Credentials not available")
            return ""




if __name__ == "__main__":
    cfg = Config()
    d = Detector(cfg.model_type, weight_path=cfg.weight_path, configs=cfg) 
    source = "../data/image"
    print(d.infer(source))