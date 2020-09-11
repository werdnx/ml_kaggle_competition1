#transfer files to remote
DEST=/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1
R_HOST=91.77.168.57
sshpass -p "V3vAndre!1988" scp -P 1022 $PWD/../../../utils/bird/*.py werdn@$R_HOST:$DEST/utils/bird/

