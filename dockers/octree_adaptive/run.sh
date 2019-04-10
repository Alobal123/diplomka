set -e
##########################################################################################################
# Set required variables

name='octree_adaptive'
dataset_path="/home/krabec/octree_adaptive_12"
out_path="/home/krabec/dockers/octree_adaptive/out/"
GPU=1
docker_hidden=d

##########################################################################################################

mkdir -p "$out_path"
docker build -t "$name" .
docker kill "$name" 2>/dev/null | true
docker rm "$name" 2>/dev/null | true

docker run --runtime=nvidia --rm -id --name "$name" -v "$out_path":/workspace/logs -v "$dataset_path":/data "$name"
docker exec -i -"$docker_hidden" "$name" sh -c "export CUDA_VISIBLE_DEVICES=$GPU && python train.py"

##########################################################################################################

if [ "$docker_hidden" == d ]; then echo Container running in detached mode. Check the log file for the information; fi
