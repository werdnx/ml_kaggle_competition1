#transfer files to remote
DEST=/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1
R_HOST=91.77.168.57
sshpass -p "V3vAndre!1988" scp -P 1022 $PWD/../../docker/melanoma/learn/Dockerfile werdn@$R_HOST:$DEST/docker/melanoma/learn
sshpass -p "V3vAndre!1988" scp -P 1022 $PWD/../../docker/melanoma/predict/Dockerfile werdn@$R_HOST:$DEST/docker/melanoma/predict
sshpass -p "V3vAndre!1988" scp -P 1022 $PWD/../../python/melanoma/* werdn@$R_HOST:$DEST/python/melanoma/

#build docker image
#DEST - path on the ubuntu f.e. /dev/sdb1/ml
sshpass -p "V3vAndre!1988" ssh -p 1022 werdn@$R_HOST 'DEST=/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1; cd $DEST/docker/melanoma/learn; docker build --tag ml_kaggle1:1.0 .'
sshpass -p "V3vAndre!1988" ssh -p 1022 werdn@$R_HOST 'DEST=/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1; cd $DEST/docker/melanoma/learn; docker rm kaggle_container'
sshpass -p "V3vAndre!1988" ssh -p 1022 werdn@$R_HOST 'DEST=/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1; cd $DEST/docker/melanoma/learn; docker run --gpus all --name kaggle_container -v /home/werdn/input:/input -v $DEST/output:/output -v $DEST/python/melanoma:/python ml_kaggle1:1.0'

