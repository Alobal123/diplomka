##########################################################################################################
# Set required variables

name='kdnet'
dataset_path="/local/krabec/ShapeNet/kdnet"
out_path="/home/krabec/dockers/kdnet/shapenet/"
GPU=2
docker_hidden=t

##########################################################################################################

mkdir "$out_path"
docker build -t "$name" .
docker kill "$name"
docker rm "$name"

docker run --runtime=nvidia --rm -id --name "$name" -v "$out_path":/kdnets/logs -v "$dataset_path":/data "$name"

docker exec -i -"$docker_hidden" "$name" sh -c "export CUDA_VISIBLE_DEVICES=$GPU && python train.py"

##########################################################################################################


#docker exec -i -"$docker_hidden" "$name" sh -c "rm -rf /kdnets/logs/*"


if [ "$docker_hidden" == d ]; then echo Container running in detached mode. Check the log file for the information; fi
