##########################################################################################################
# Set required variables
set -e
name="shaded"
dataset_path="/local/krabec/ShapeNet/ShapeNet"
output_dir="/local/krabec/ShapeNet"
docker_hidden=d

#This must be one of phong, shaded or depth
render=shaded

##########################################################################################################

image_name="blender"
output_dir="$output_dir/$name"
mkdir -p "$output_dir"
docker build -t "$image_name" .
docker kill "$image_name" 2>/dev/null | true
docker rm "$image_name" 2>/dev/null | true

docker run --rm -id --name "$image_name" -v "$dataset_path":/dataset -v "$output_dir":/data "$image_name"
docker exec -i -"$docker_hidden" "$image_name" sh -c "/usr/local/blender/blender render_$render.blend -noaudio -b -P  blender_data.py" 

##########################################################################################################

if [ "$docker_hidden" == d ]; then echo Container running in detached mode. Check the log file for the information; fi
