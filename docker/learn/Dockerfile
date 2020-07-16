FROM tensorflow/tensorflow:latest-gpu
WORKDIR /python
RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip install opencv-python
RUN pip install keras
RUN pip install tqdm
RUN pip install -U scikit-learn scipy matplotlib

CMD ["python","-u","learn.py"]
