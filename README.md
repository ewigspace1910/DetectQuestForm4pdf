# Detect Question forms from pdf
 
![fig](./data/overview.png)



# Getting 
- Evironment :

```python
!pip install ultralytics #for Yolov8
!pip install opencv-python
!pip install torch
```

## Training
- [yolov8](notebook/README.md)



## Deploy
- Create Restful API by fastAPI
  - run server
    ```python
    python deloy/app.py
    ```
  - test docs: "localhost:9000/docs"
- Build Docker image and test:
  - `docker build . -t test`
  - `docker run -p 9000:9000 test:latest`
  - - test docs: "localhost:9000/docs"

- Deploy on AWS Lambda [tutorial](https://youtu.be/VYk3lwZbHBU)
  - login to aws
  - build docker : ```docker build -t test_layout_analysis ``` 
  - put on aws  