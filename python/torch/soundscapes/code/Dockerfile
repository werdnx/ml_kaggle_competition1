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
RUN pip install audiomentations
RUN pip install -e git+https://github.com/ncullen93/torchsample.git#egg=torchsample
RUN mkdir /work
COPY . /work
WORKDIR /work
# Again, test and train scripts should be executable within the container.
RUN chmod +x test.sh
RUN chmod +x train.sh
RUN chmod +x cnv.sh
# Don't add any CMD or ENTRYPOINT!