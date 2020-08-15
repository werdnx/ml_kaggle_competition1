#transfer files to remote
R_HOST=91.77.168.57
DEST=/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1
sshpass -p "V3vAndre!1988" scp -P 1022 $PWD/../../utils/* werdn@$R_HOST:$DEST/utils/

