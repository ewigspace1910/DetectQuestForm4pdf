import os
from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

# from deploy.routers import router_analysis
from routers import router_analysis

app=FastAPI()



###############################################################################
#   Routers configuration                                                     #
###############################################################################

@app.get("/")
async def root():
    return {"message": "/docs to read API documents"}

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
