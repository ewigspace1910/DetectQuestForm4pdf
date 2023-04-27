#Thirst Library
from ultralytics import YOLO
import json
# DETECTION
from layoutdetection.infer_yolov8 import infer_yolov8

import cv2
import io

#OCR
from ocr import mathpix_ocrrequest



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

### auxillary function
def combine_min_elements(orders, elements):
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

class Detector():
    def __init__(self, model, weight_path):
        """
        model : name of model type you want to infer. Nows, we supports: yolov8
        weight_path: path to model/weight file : .pt, onnx. Depend on the type of package, you need to carefully pass the value to weight_path.
        """
        self.infer_func = self.model = None
        if model == "yolov8":
            self.model = YOLO(weight_path)
            self.infer_func = infer_yolov8

    def infer(self, inputs):
        """
        inputs: current types we supports are paths to images
        """
        detection_results = self.infer_func(source=inputs, model=self.model)

        elements = []
        orders   = []

        def save_image(img):
            name = "xxx.png"
            cv2.imwrite(name, img)
            return name

        for i, result in enumerate(detection_results):
            for bidx, box in enumerate(result.boxes):
                cls = REVERSE_MAP[int(box.cls)]
                bb = box.xyxy[0].numpy()
                cropped_image = detection_results[i].orig_img[int(bb[1]): int(bb[3]), int(bb[0]): int(bb[2])]
                text = ""
                if cls in ["image", "table"]: #save image/table
                    path = save_image(cropped_image)
                else: #OCR on text only: question, subquestion, choice(text), blank, auxillary_text
                    _, buffer = cv2.imencode(".png", cropped_image)
                    io_buffer = io.BytesIO(buffer)
                    text = mathpix_ocrrequest(io_buffer=io_buffer)

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
    
    def __group_element(self, orders, elements):
        """Args:
            - orders : (list[int])   Contain category_weights corresponding elements
            - elements: (list[dict]) Contain all detected element in order from top to bottom of page, from page 0 to page n
            Return: json object like html document object
        """
        o, e = orders, elements
        while len(set(o)) > 1:
            o, e = combine_min_elements(o,e)
        return e

if __name__ == "__main__":
    d = Detector("yolov8", weight_path="../data/yolov8M.pt") 
    source = "../data/image"
    print(d.infer(source))