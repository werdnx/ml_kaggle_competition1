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
RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/ufoym/imbalanced-dataset-sampler.git imbalanced-dataset-sampler
RUN cd imbalanced-dataset-sampler
RUN python3 setup.py install
RUN pip3 install .
RUN mkdir /work
COPY . /work
WORKDIR /work
# Again, test and train scripts should be executable within the container.
RUN chmod +x test.sh
RUN chmod +x train.sh
RUN chmod +x cnv.sh
# Don't add any CMD or ENTRYPOINT!