set -e 
##########################################################################################################
# Set required variables

name='vrnens'
dataset_path="/path/to/dataset"
out_path="/output/path"
GPU=0
docker_hidden=t

##########################################################################################################

mkdir  "$out_path"
docker build -t "$name" .
docker kill "$name" 2>/dev/null | true
docker rm "$name" 2>/dev/null | true

docker run --runtime=nvidia --rm -id --name "$name" -v "$out_path":/vrnens/Discriminative/logs -v "$dataset_path":/data "$name"

docker exec -i -"$docker_hidden" "$name" sh -c "export CUDA_VISIBLE_DEVICES=$GPU && python train.py > /vrnens/Discriminative/logs/log.txt"

##########################################################################################################

if [ "$docker_hidden" == d ]; then echo Container running in detached mode. Check the log file for the information; fi
