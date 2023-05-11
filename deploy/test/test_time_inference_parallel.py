import time
import numpy as np
from PIL import Image
import os 
import sys 
import time
sys.path.append(os.getcwd())
from deploy.detector import get_detector
import json

# @profile
def time_detector_infer(objs):
    Detector =  get_detector()
    # layout = Detector.infer_parallel(objs)
    layout = Detector.infer_concurrent(objs)
    with open("layout.json", "w") as outfile:
        json.dump(layout, outfile, indent=4)


if __name__ == '__main__':
    paths = ["deploy/test/scanned-{}.png".format(i) for i in range(9)]
    # paths = ["deploy/test/scanned-{}.png".format(2)]
    imgs = [np.array(Image.open(p)) for p in paths]
    for _ in range(1):
        start_time = time.perf_counter()
        time_detector_infer(imgs)
        print("total execution time -->{:.5f} s".format(time.perf_counter() - start_time))
    #run kernprof -lv .\deploy\test\test_time_inference_parallel.py