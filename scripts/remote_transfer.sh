#transfer files to remote
DEST=/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1
R_HOST=192.168.1.78
sshpass -p "werdn" scp $PWD/../docker/Dockerfile werdn@$R_HOST:$DEST/docker/
sshpass -p "werdn" scp $PWD/../python/* werdn@$R_HOST:$DEST/python/

