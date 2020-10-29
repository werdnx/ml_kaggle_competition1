#!/bin/bash
echo "convertation dir is $(pwd)"
for i in *.flac; do
  name=$(echo "$i" | cut -d'.' -f1)
  echo "processing $name"
  ffmpeg -i "$i" "${1}/${name}.wav"
done
