#transfer files to remote
DEST=/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1
R_HOST=91.77.168.57
sshpass -p "V3vAndre!1988" scp -P 1022 $PWD/../../../docker/torch_audio/Dockerfile werdn@$R_HOST:$DEST/docker/torch_audio
sshpass -p "V3vAndre!1988" scp -P 1022 $PWD/../../../python/torch/audio/* werdn@$R_HOST:$DEST/python/torch/audio

#build docker image
#DEST - path on the ubuntu f.e. /dev/sdb1/ml
sshpass -p "V3vAndre!1988" ssh -p 1022 werdn@$R_HOST 'DEST=/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1; cd $DEST/docker/torch_audio; docker build --tag ml_torch1:1.0 .'
sshpass -p "V3vAndre!1988" ssh -p 1022 werdn@$R_HOST 'DEST=/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1; cd $DEST/docker/torch_audio; docker rm torch_container'
sshpass -p "V3vAndre!1988" ssh -p 1022 werdn@$R_HOST 'DEST=/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1; cd $DEST/docker/torch_audio; docker run --gpus all --name torch_container -v $DEST/input/torch/audio:/input -v $DEST/output/torch/audio:/output -v $DEST/python/torch/audio:/python ml_torch1:1.0'

