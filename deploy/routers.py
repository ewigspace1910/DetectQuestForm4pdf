from fastapi.responses import JSONResponse
from fastapi import APIRouter, File, UploadFile, FastAPI
import fitz
import os
import time
from typing import List
from PIL import Image
import io

# from deploy.detector import get_detector, CATEGORY_MAP
from detector import get_detector,  CATEGORY_MAP
# from detector_onnx import get_detector,  CATEGORY_MAP

DETECTOR = get_detector()

router_analysis = APIRouter(
    prefix="/analysis",
    tags=['PDF Layout Analysis']
)

@router_analysis.get("/labels")
def get_info_api():
    return JSONResponse({
        "CATEGORY_MAP" : CATEGORY_MAP
    })

# @router_analysis.post("/pdf")
# async def post_pdf_file(pdf:UploadFile=File(...)):
#     tstart = time.time()
#     if pdf.filename.lower().find(".pdf") < 0: return JSONResponse({"error": "File upload must be .pdf"})
#     layout = {}


#     #save file pdf from client
#     rand_name = pdf.filename.replace(" ", "_")
#     with open(rand_name, "wb") as f:
#         f.write(await pdf.read())
   

#     #convert pdf2image and save for prediction
#     zoom_x, zoom_y = 2.0, 2.0
#     mat = fitz.Matrix(zoom_x, zoom_y)  
#     doc = fitz.open(rand_name)
#     img_list = []
#     for i, page in enumerate(doc):  
#         pix = page.get_pixmap(matrix=mat) 
#         iname = f"{rand_name}_{page.number}.png"
#         pix.save(iname)
#         img_list.append(iname)
#         if i == 6: break
#     doc.close()
    
#     # detection
#     Detector =  DETECTOR
#     # layout = Detector.infer(img_list)
#     layout = Detector.infer_concurrent(img_list)

#     # clean tmp files
#     for x in img_list: os.remove(x)
#     os.remove(rand_name)

#     return JSONResponse({
#         "success" : True,
#         "filename": pdf.filename,
#         "layout": layout,
#         "excution time" : time.time() - tstart
#     })


@router_analysis.post("/layout")
async def post_imgs(images:List[UploadFile]=File(...)):
    tstart = time.time()
    try:
        objs = []
        for img in images:
            contents = await img.read()
            objs += [Image.open(io.BytesIO(contents)).convert("RGB")]

        Detector =  DETECTOR
        layout = Detector.infer(objs)

        res = JSONResponse({
            "status": "success",
            "data": layout,
            "excution time" : time.time() - tstart
        })
        return res
    except Exception as e:
        print(e)
        return JSONResponse({
            "status": "false",
            "data": {},
            "excution time" : time.time() - tstart,
            "error": e
        })

@router_analysis.post("/layout/speedup")
async def post_imgs_parallel(images:List[UploadFile]=File(...)):
    tstart = time.time()
    try:
        objs = []
        for img in images:
            contents = await img.read()
            objs += [Image.open(io.BytesIO(contents)).convert("RGB")]

        Detector =  Detector =  DETECTOR
        # layout = Detector.infer_parallel(objs)
        layout = Detector.infer_concurrent(objs)

        res = JSONResponse({
            "status": "success",
            "data": layout,
            "excution time" : time.time() - tstart
        })
        return res
    except Exception as e:
        print(e)
        return JSONResponse({
            "status": "false",
            "data": {},
            "excution time" : time.time() - tstart,
            "error": e
        })
    

    