FROM tensorflow/tensorflow:nightly-gpu
WORKDIR /python
RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip install opencv-python
RUN pip install keras
RUN pip install tqdm
RUN pip install -U scikit-learn scipy matplotlib
#RUN pip install -U efficientnet
RUN pip install pandas
RUN pip install tensorflow_datasets

CMD ["python","-u","predict2.py"]
