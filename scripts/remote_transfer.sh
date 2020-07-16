#transfer files to remote
DEST=/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1
R_HOST=192.168.1.77
sshpass -p "werdn" scp $PWD/../docker/learn/Dockerfile werdn@$R_HOST:$DEST/docker/learn
sshpass -p "werdn" scp $PWD/../docker/predict/Dockerfile werdn@$R_HOST:$DEST/docker/predict
sshpass -p "werdn" scp $PWD/../python/* werdn@$R_HOST:$DEST/python/

