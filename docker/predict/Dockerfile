FROM tensorflow/tensorflow:1.5.0-gpu
WORKDIR /python
RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip install opencv-python
RUN pip install keras==2.1.4
RUN pip install tqdm

CMD ["python","-u","predict.py"]
