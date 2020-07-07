#build docker image
#DEST - path on the ubuntu f.e. /dev/sdb1/ml
R_HOST=192.168.1.78
sshpass -p "werdn" ssh werdn@$R_HOST 'DEST=/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1; cd $DEST/docker; docker build --tag ml_kaggle1:1.0 .'
sshpass -p "werdn" ssh werdn@$R_HOST 'DEST=/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1; cd $DEST/docker; docker rm kaggle_container'
sshpass -p "werdn" ssh werdn@$R_HOST 'DEST=/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1; cd $DEST/docker; docker run --gpus all --name kaggle_container -v $DEST/input:/input -v $DEST/output:/output -v $DEST/python:/python ml_kaggle1:1.0'