#transfer files to remote
DEST=/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1
R_HOST=91.77.168.57
sshpass -p "V3vAndre!1988" scp -P 1022 $PWD/../../../docker/city/learn/Dockerfile werdn@$R_HOST:$DEST/docker/city/learn
sshpass -p "V3vAndre!1988" scp -P 1022 $PWD/../../../python/city/* werdn@$R_HOST:$DEST/python/city/

#build docker image
#DEST - path on the ubuntu f.e. /dev/sdb1/ml
sshpass -p "V3vAndre!1988" ssh -p 1022 werdn@$R_HOST 'DEST=/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1; cd $DEST/docker/city/learn; docker build --tag ml_city1:1.0 .'
sshpass -p "V3vAndre!1988" ssh -p 1022 werdn@$R_HOST 'DEST=/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1; cd $DEST/docker/city/learn; docker rm city_container'
sshpass -p "V3vAndre!1988" ssh -p 1022 werdn@$R_HOST 'DEST=/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1; cd $DEST/docker/city/learn; docker run --gpus all --name city_container -v /media/3tstor/ml/torch/soundscapes/data:/data -v /media/3tstor/ml/torch/soundscapes/wdata:/wdata -v $DEST/python/city:/python ml_city1:1.0'

