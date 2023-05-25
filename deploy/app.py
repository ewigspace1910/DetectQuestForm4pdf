import os
from fastapi import FastAPI, Request
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

# from deploy.routers import router_analysis, DETECTOR
from routers import router_analysis, DETECTOR

app=FastAPI()



###############################################################################
#   Routers configuration                                                     #
###############################################################################

@app.get("/")
async def root(request: Request):
    return {"message":  f"experiment API in: {str(request.url)}docs",
            "Machine configs-CPU": os.cpu_count(),
            "Model ready!": DETECTOR.ready()}

app.include_router(router_analysis)

###############################################################################
#   Handler for AWS Lambda                                                    #
###############################################################################

handler = Mangum(app)

###############################################################################
#   Run the self contained application                                        #
###############################################################################

if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=9000)
