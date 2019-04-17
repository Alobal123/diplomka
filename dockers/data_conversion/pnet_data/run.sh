##########################################################################################################
# Set required variables
set -e
name="lloyd"
dataset_path="/local/krabec/ModelNet40A"
output_dir="/home/krabec/"
docker_hidden=t

##########################################################################################################

image_name="pointcloud"

output_dir="$output_dir/$name"
mkdir -p "$output_dir"
docker build -t "$image_name" .
docker kill "$image_name" 2>/dev/null | true
docker rm "$image_name" 2>/dev/null | true

docker run --rm -id --name "$image_name" -v "$dataset_path":/dataset -v "$output_dir":/data "$image_name"
docker exec -i -"$docker_hidden" "$image_name" sh -c "python3 pnet_data.py"

##########################################################################################################

if [ "$docker_hidden" == d ]; then echo Container running in detached mode. Check the log file for the information; fi
