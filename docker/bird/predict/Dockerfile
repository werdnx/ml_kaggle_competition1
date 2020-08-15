FROM tensorflow/tensorflow:2.3.0-gpu
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
RUN pip install librosa

CMD ["python","-u","-W ignore","main_predict.py"]
