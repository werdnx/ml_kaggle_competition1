DEST=/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1
cd $DEST/docker/learn; docker build --tag ml_kaggle1:1.0 .
docker rm kaggle_container
docker run --gpus all --name kaggle_container -v /home/werdn/input:/input -v $DEST/output:/output -v $DEST/python:/python ml_kaggle1:1.0
