#Thirst Library
from ultralytics import YOLO
import multiprocessing as mp
import json
# import cv2
import numpy as np
import io
import requests

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
        
        #model
        self.model_type = "yolov8"
        self.weight_path = "deploy/model/yolov8.pt"

        #parallel
        self.n_processes = 4

    
##############################
##
## element modules
##############################
#yolov8
def infer_yolov8(source, model):
    """
    source:  could be image, path,... 
    """
    detection_results = []
    detection_results = model.predict(source, save=False, imgsz=(640, 640), conf=0.45, device="cpu")
    return detection_results


#mathpix ocr function
def mathpix_ocrrequest(io_buffer, app_id="", app_key=""):
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

def save_image(img):
    name = "linkimage.png"
    #save image and getlink
    # cv2.imwrite(name, img)
    return name

#########################################
### 
##       PIPELINE
###
#########################################
def get_detector():
    detector = Detector(configs=Config())
    return detector

def subprocess_ocr_request(boxes, orig_imgs, app_id, app_key, conn):
    """
    boxes: (page index, detectedbox) list from yolov8 results
    orgin_img : (numpy array): original imaege
    conn     : multiprocessing Pipe to receive results 
    """
    objs =[]
    for page_index, box in boxes:
        cls = REVERSE_MAP[int(box.cls)]
        bb = box.xyxy[0].numpy()
        cropped_image = orig_imgs[page_index].orig_img[int(bb[1]): int(bb[3]), int(bb[0]): int(bb[2])]
        text = ""
        if cls in ["image", "table"]: #save image/table
            text = save_image(cropped_image)
        else:
            io_buffer = io.BytesIO()
            np.save(io_buffer, cropped_image)
            io_buffer.seek(0)
            text = mathpix_ocrrequest(io_buffer=io_buffer, app_id= app_id, app_key=app_key)

        obj = {"name":cls, 
                "text": text,
                'child':[], 
                "idx": (page_index, int(box.xyxy[0][1]))} #(page number, y-position)
        objs += [obj]

    conn.send(objs)
    conn.close()

class Detector(object):
    def __init__(self, configs:Config):
        self.infer_func = self.model = None
        self.cfg = configs

        if self.cfg.model_type == "yolov8":
            self.model = YOLO(self.cfg.weight_path)
            self.infer_func = infer_yolov8

    def infer(self, inputs):
        """
        inputs: upload file from http requestion

        Return --> list(dict) : list of question objects with their child elements
        """
        #Prediction
        detection_results = self.infer_func(source=inputs, model=self.model)

        elements = []
        orders   = []
        for i, result in enumerate(detection_results):
            for bidx, box in enumerate(result.boxes):
                cls = REVERSE_MAP[int(box.cls)]
                bb = box.xyxy[0].numpy()
                cropped_image = detection_results[i].orig_img[int(bb[1]): int(bb[3]), int(bb[0]): int(bb[2])]
                text = ""
                if cls in ["image", "table"]: #save image/table
                    text = save_image(cropped_image)
                else: #OCR on text only: question, subquestion, choice(text), blank, auxillary_text
                    # _, buffer = cv2.imencode(".png", cropped_image)
                    # io_buffer = io.BytesIO(buffer)
                    io_buffer = io.BytesIO()
                    np.save(io_buffer, cropped_image)
                    io_buffer.seek(0)
                    text = mathpix_ocrrequest(io_buffer=io_buffer, app_id=self.cfg.matpix_id, app_key=self.cfg.matpix_key)

                obj = {"name":cls, 
                        "text": text,
                    'child':[], 
                    "idx": (i, int(box.xyxy[0][1]))} #(page number, y-position)
                elements += [obj]
                
        elements = sorted(elements, key=lambda x: x["idx"])
        orders = [CATEGORY_LEVEL[obj['name']] for obj in elements]
        if orders[0] < 3:
            orders = [3] + orders
            elements = [{"name":'heading', 
                        "text": "Exam Questions",
                    'child':[], 
                    "idx": (0, 0)}] + elements
        
        #GROUP detected elements to questions.
        questions = self.__group_element(orders, elements)

        return questions #convert to Json???/
    
    def infer_parallel(self, inputs):
        """
        inputs: upload file from http requestion

        Return --> list(dict) : list of question objects with their child elements
        """
        #Prediction
        detection_results = self.infer_func(source=inputs, model=self.model)

        #OCR
        processes = [] 
        parent_connections = []
        boxes = [(page_index, box) for page_index, results in enumerate(detection_results) for box in results.boxes]
        elements_per_process = len(boxes) // (self.cfg.n_processes)
        boxes = [boxes[i:i+ elements_per_process] for i in range(0, len(boxes), elements_per_process)]
        for b in boxes:
            parent_conn, child_conn = mp.Pipe()
            process = mp.Process(target=subprocess_ocr_request, args=(b, detection_results, self.cfg.matpix_id, self.cfg.matpix_key ,
                                                                    child_conn, ))
            process.start()
            processes += [process]
            parent_connections += [parent_conn]

        # for p in processes: p.start()
        for p in processes: p.join()
        
        elements = [obj for parent_connection in parent_connections for obj in parent_connection.recv()]
        #Postprocesing
        elements = sorted(elements, key=lambda x: x["idx"])
        orders = [CATEGORY_LEVEL[obj['name']] for obj in elements]
        if orders[0] < 3:
            orders = [3] + orders
            elements = [{"name":'heading', 
                        "text": "Exam Questions",
                        'child':[], 
                        "idx": (0, 0)}] + elements
        
        #GROUP detected elements to questions.
        questions = self.__group_element(orders, elements)

        return questions #convert to Json???/

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
                        elements[new_arr_e[i-1]]["child"] += [ elements[j] ] 
                i += 1

            new_arr_2_e = [elements[i] for i in new_arr_2_e]

            return new_arr_2, new_arr_2_e

        #run
        o, e = orders, elements
        while len(set(o)) > 1:
            o, e = __combine_min_elements(o,e)
        return e



if __name__ == "__main__":
    cfg = Config()
    d = Detector(cfg.model_type, weight_path=cfg.weight_path, configs=cfg) 
    source = "../data/image"
    print(d.infer(source))