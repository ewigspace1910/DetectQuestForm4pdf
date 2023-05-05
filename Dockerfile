FROM python:3.8

RUN mkdir -p /app /tmp
WORKDIR /app 

# RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

#setup folder
COPY ./deploy        ./deploy
COPY ./deploy/app.py .
# initialize environment
COPY ./requirements.txt ./requirements.txt 
RUN pip install -r ./requirements.txt --no-cache
RUN pip install opencv-python-headless

ENV HOST="0.0.0.0"
ENV PORT=9000
ENTRYPOINT uvicorn app:app --host ${HOST} --port ${PORT}
