#!/bin/bash
#"$1" is data folder
cur_dir=$(pwd)
echo "pwd is "
echo "$cur_dir"
if [ "$3" = "1" ]; then
  echo "not to do convertation"
else
  echo "do convertation"
  cd "$1"
  mkdir /wdata/test
  bash "${cur_dir}/cnv.sh" /wdata/test
  cd "$cur_dir"
fi
if [ "$4" = "1" ]; then
  echo "not to do audio preprocessing"
else
  echo "do test audio pre processing"
  mkdir /wdata/preprocessed_audio_test
  python3 ./src/preprocess_audio_test.py "$1" "$2"
fi

python3 ./src/test.py "$1" "$2"