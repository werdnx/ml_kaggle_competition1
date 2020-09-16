FROM tensorflow/tensorflow:2.3.0-gpu
WORKDIR /python
RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip install opencv-python
RUN pip install keras
RUN pip install tqdm
RUN pip install pandas
RUN pip install tensorflow_datasets
RUN pip install matplotlib
RUN pip install -U scikit-learn==0.23.2
RUN pip install xgboost==1.2.0
RUN pip install imbalanced-learn
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
RUN pip install patsy
RUN pip install statsmodels
RUN pip install git+https://github.com/scikit-learn-contrib/categorical-encoding.git
RUN pip install tensorflow-addons

CMD ["python","-u","netwok_learn2.py"]
