#transfer files to remote
DEST=/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1
R_HOST=91.77.168.57
sshpass -p "V3vAndre!1988" scp -P 1022 $PWD/../../docker/learn/Dockerfile werdn@$R_HOST:$DEST/docker/learn
sshpass -p "V3vAndre!1988" scp -P 1022 $PWD/../../docker/predict/Dockerfile werdn@$R_HOST:$DEST/docker/predict
sshpass -p "V3vAndre!1988" scp -P 1022 $PWD/../../python/* werdn@$R_HOST:$DEST/python/

#build docker image
#DEST - path on the ubuntu f.e. /dev/sdb1/ml
sshpass -p "V3vAndre!1988" ssh -p 1022 werdn@$R_HOST 'DEST=/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1; cd $DEST/docker/predict; docker build --tag ml_kaggle_predict1:1.0 .'
sshpass -p "V3vAndre!1988" ssh -p 1022 werdn@$R_HOST 'DEST=/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1; cd $DEST/docker/predict; docker rm kaggle_container_predict'
sshpass -p "V3vAndre!1988" ssh -p 1022 werdn@$R_HOST 'DEST=/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1; cd $DEST/docker/predict; docker run --gpus all --name kaggle_container_predict -v /home/werdn/input:/input -v $DEST/output:/output -v $DEST/python:/python ml_kaggle_predict1:1.0'