#transfer files to remote
DEST=/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1
R_HOST=91.77.168.57
sshpass -p "V3vAndre!1988" scp -P 1022 $PWD/../../../docker/moa/learn/Dockerfile werdn@$R_HOST:$DEST/docker/moa/learn
#sshpass -p "V3vAndre!1988" scp -P 1022 $PWD/../../../docker/moa/predict/Dockerfile werdn@$R_HOST:$DEST/docker/moa/predict
sshpass -p "V3vAndre!1988" scp -P 1022 $PWD/../../../python/moa/* werdn@$R_HOST:$DEST/python/moa/

#build docker image
#DEST - path on the ubuntu f.e. /dev/sdb1/ml
sshpass -p "V3vAndre!1988" ssh -p 1022 werdn@$R_HOST 'DEST=/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1; cd $DEST/docker/moa/learn; docker build --tag ml_kaggle1:1.0 .'
sshpass -p "V3vAndre!1988" ssh -p 1022 werdn@$R_HOST 'DEST=/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1; cd $DEST/docker/moa/learn; docker rm kaggle_container'
sshpass -p "V3vAndre!1988" ssh -p 1022 werdn@$R_HOST 'DEST=/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1; cd $DEST/docker/moa/learn; docker run --gpus all --name kaggle_container -v /home/werdn/input/moa:/input -v $DEST/output/moa:/output -v $DEST/python/moa:/python ml_kaggle1:1.0'

