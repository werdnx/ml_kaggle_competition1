#build docker image
#DEST - path on the ubuntu f.e. /dev/sdb1/ml
R_HOST=192.168.1.77
sshpass -p "V3vAndre!1988" ssh werdn@$R_HOST 'DEST=/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1; cd $DEST/docker/learn; docker build --tag ml_kaggle1:1.0 .'
sshpass -p "V3vAndre!1988" ssh werdn@$R_HOST 'DEST=/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1; cd $DEST/docker/learn; docker rm kaggle_container'
sshpass -p "V3vAndre!1988" ssh werdn@$R_HOST 'DEST=/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1; cd $DEST/docker/learn; docker run --gpus all --name kaggle_container -v /home/werdn/input:/input -v $DEST/output:/output -v $DEST/python:/python ml_kaggle1:1.0'