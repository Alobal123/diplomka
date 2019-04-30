set -e

##########################################################################################################
# Set required variables

name='kdnet2'
#dataset_path="/local/krabec/ModelNet40A/kdnet"
#out_path="/home/krabec/dockers/kdnet/out3"
dataset_path="/local/krabec/ShapeNet/kdnet"
out_path="/home/krabec/dockers/kdnet/shapenet"
GPU=1
docker_hidden=t

##########################################################################################################

mkdir -p "$out_path"
docker build -t "$name" .
docker kill "$name" 2>/dev/null | true
docker rm "$name" 2>/dev/null | true
docker run --runtime=nvidia --rm -id --name "$name" -v "$out_path":/kdnets/logs -v "$dataset_path":/data "$name"
docker exec -i -"$docker_hidden" "$name" sh -c "export CUDA_VISIBLE_DEVICES=$GPU && python train.py"
##########################################################################################################
