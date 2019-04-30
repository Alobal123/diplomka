set -e

##########################################################################################################
# Set required variables

name='mvcnn'
dataset_path="/path/to/dataset"
out_path="/output/path"
GPU=0
docker_hidden=t

##########################################################################################################

mkdir -p "$out_path"
docker build -t "$name" .
docker kill "$name" 2>/dev/null | true
docker rm "$name" 2>/dev/null | true
docker run --runtime=nvidia --rm -id --name "$name" -v "$out_path":/mvcnn/logs -v "$dataset_path":/data "$name"
docker exec -i -"$docker_hidden" "$name" sh -c "export CUDA_VISIBLE_DEVICES=$GPU && python train.py"
##########################################################################################################
