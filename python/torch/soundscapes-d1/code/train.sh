#!/bin/bash
#"$1" is data folder
cur_dir=$(pwd)
echo "pwd is "
echo "$cur_dir"
mkdir /wdata/model
if [ "$2" = "1" ]; then
  echo "not to do convertation"
else
  echo "do convertation"
  cd "$1"
  mkdir /wdata/train
  bash "${cur_dir}/cnv.sh" /wdata/train
  cd "$cur_dir"
fi
if [ "$3" = "1" ]; then
  echo "not to do audio preprocessing"
else
  echo "do audio pre processing"
  mkdir /wdata/preprocessed_audio
  python3 ./src/preprocess_audio.py "$1"
fi

python3 ./src/train.py "$1"