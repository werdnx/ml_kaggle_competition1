FROM tensorflow/tensorflow:2.1.0-gpu-py3
WORKDIR /python
RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip install opencv-python
RUN pip install keras==2.3.1
RUN pip install tqdm
RUN pip install -U scikit-learn scipy matplotlib
RUN pip install -U efficientnet
RUN pip install pandas

CMD ["python","-u","learn.py"]
