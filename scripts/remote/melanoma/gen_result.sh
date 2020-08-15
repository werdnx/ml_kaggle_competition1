#transfer files to remote
R_HOST=91.77.168.57
DEST=/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1
sshpass -p "V3vAndre!1988" scp -P 1022 $PWD/../../python/melanoma/* werdn@$R_HOST:$DEST/python/melanoma/
sshpass -p "V3vAndre!1988" ssh -p 1022 werdn@$R_HOST 'DEST=/media/3tstor/ml/IdeaProjects/ml_kaggle_competition1; cd $DEST/python/melanoma; python gen_result.py'
sshpass -p "V3vAndre!1988" scp -P 1022 werdn@$R_HOST:$DEST/output/result* /Users/dmitrenkoandrey/PycharmProjects/ml_kaggle_competition1/result/

