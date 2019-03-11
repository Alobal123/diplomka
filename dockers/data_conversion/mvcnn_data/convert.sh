##########################################################################################################
# Set required variables

#name="ModelNet40A_mvcnn_pbrt"
name="Small_converted"
#dataset="/home/krabec/data/ModelNet40A"
dataset="/local/krabec/Small"
output_dir="/local/krabec"
docker_hidden=t

##########################################################################################################

image_name="pbrt"

output_dir="$output_dir/$name"
mkdir -m 777 $output_dir
docker build -t "$image_name" .
docker kill "$image_name"
docker rm "$image_name"

docker run --rm -id --name "$image_name" -v "$dataset":/dataset -v "$output_dir":/data "$image_name"
docker exec -i -"$docker_hidden" "$image_name" sh -c "python3 mvcnn_data.py"

##########################################################################################################
