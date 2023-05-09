from fastapi.responses import JSONResponse
from fastapi import APIRouter, File, UploadFile
import fitz
import os
import time
from typing import List
from PIL import Image
import io

# from deploy.detector import get_detector, Config, CATEGORY_MAP
from detector import get_detector, Config, CATEGORY_MAP

router_analysis = APIRouter(
    prefix="/analysis",
    tags=['PDF Layout Analysis']
)

@router_analysis.get("/labels")
def get_info_api():
    cfg = Config()
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
#     for page in doc:  
#         pix = page.get_pixmap(matrix=mat) 
#         iname = f"{rand_name}_{page.number}.png"
#         pix.save(iname)
#         img_list.append(iname)
#     doc.close()
    
#     # detection
#     try:
#         Detector =  get_detector()
#         layout = Detector.infer(img_list)
#     except Exception as e: 
#         layout = "DETECTOR GET ERROR!!!!-->" + str(e)

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
    layout = {}
    try:
        objs = []
        for img in images:
            contents = await img.read()
            objs += [Image.open(io.BytesIO(contents))]

        Detector =  get_detector()
        # layout = Detector.infer(objs)
        layout = Detector.infer_parallel(objs)

        return JSONResponse({
            "status": "success",
            "data": layout,
            "excution time" : time.time() - tstart
        })
    except Exception as e:
        return JSONResponse({
            "status": "false",
            "data": {},
            "excution time" : time.time() - tstart,
            "error": e
        })