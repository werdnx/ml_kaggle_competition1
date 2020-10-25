FROM pytorch/pytorch
WORKDIR /python
RUN apt-get update
RUN pip install pandas
RUN pip install librosa
RUN pip install soundfile
RUN apt-get install -y libsndfile1
RUN pip install seaborn
RUN apt-get install -y ffmpeg
RUN pip install noisereduce
RUN pip install albumentations

CMD ["python","-u","train.py"]
