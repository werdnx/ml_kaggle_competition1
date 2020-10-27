#transfer files to remote
DEST=/media/3tstor/ml/torch
R_HOST=91.77.168.57
sshpass -p "V3vAndre!1988" scp -rP 1022 $PWD/../../../python/torch/soundscapes/* werdn@$R_HOST:$DEST/soundscapes/app
#/data/train
#/data/test
#sshpass -p "V3vAndre!1988" ssh -p 1022 werdn@$R_HOST 'DEST=/media/3tstor/ml/torch; cd $DEST/soundscapes/app/code; docker build -t ml_torch1:1.0 .'
#sshpass -p "V3vAndre!1988" ssh -p 1022 werdn@$R_HOST 'DEST=/media/3tstor/ml/torch; cd $DEST/soundscapes/app/code; docker rm torch_container'
#sshpass -p "V3vAndre!1988" ssh -p 1022 werdn@$R_HOST 'DEST=/media/3tstor/ml/torch; cd $DEST/soundscapes/app/code; docker run --gpus all --name torch_container -v $DEST/soundscapes/data:/data:ro -v $DEST/soundscapes/wdata:/wdata -it ml_torch1:1.0'

