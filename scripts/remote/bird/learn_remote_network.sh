#transfer files to remote
DEST=/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1
R_HOST=91.77.168.57
sshpass -p "V3vAndre!1988" scp -P 1022 $PWD/../../../docker/bird/learn/Dockerfile werdn@$R_HOST:$DEST/docker/bird/learn
sshpass -p "V3vAndre!1988" scp -P 1022 $PWD/../../../docker/bird/predict/Dockerfile werdn@$R_HOST:$DEST/docker/bird/predict
sshpass -p "V3vAndre!1988" scp -P 1022 $PWD/../../../python/bird/* werdn@$R_HOST:$DEST/python/bird/

#build docker image
#DEST - path on the ubuntu f.e. /dev/sdb1/ml
sshpass -p "V3vAndre!1988" ssh -p 1022 werdn@$R_HOST 'DEST=/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1; cd $DEST/docker/bird/learn; docker build --tag ml_kaggle1:1.0 .'
sshpass -p "V3vAndre!1988" ssh -p 1022 werdn@$R_HOST 'DEST=/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1; cd $DEST/docker/bird/learn; docker rm kaggle_container'
sshpass -p "V3vAndre!1988" ssh -p 1022 werdn@$R_HOST 'DEST=/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1; cd $DEST/docker/bird/learn; docker run --gpus all --name kaggle_container -v /home/werdn/input/bird:/input -v $DEST/output/bird:/output -v $DEST/python/bird:/python ml_kaggle1:1.0'

