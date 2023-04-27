
def infer_yolov8(source, model):
    """
    source:  could be image, path,... 
    """
    detection_results = model.predict(source, save=True, imgsz=(640, 640), conf=0.4, device="cpu")
    return detection_results