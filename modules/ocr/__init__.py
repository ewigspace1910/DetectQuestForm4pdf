import requests
import json
import io

#@title mathpix ocr function
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