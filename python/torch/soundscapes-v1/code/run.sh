docker build -t ml_torch1:1.0 .
docker rm torch_container
docker run --gpus all --name torch_container -v /media/3tstor/ml/torch/soundscapes/data:/data:ro -v /media/3tstor/ml/torch/soundscapes/wdata:/wdata -it ml_torch1:1.0

