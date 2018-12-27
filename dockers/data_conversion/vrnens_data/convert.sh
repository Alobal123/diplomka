docker build -t openvdb .

#Path to the dataset
dataset="/home/krabec/data/ShapeNet"
#Type of the dataset. Currently must be one of (modelnet, shapenet)
dataset_type="shapenet"

#To see other options run python script with -h option and change docker run parameter to -d to -it.
docker run --rm -d --mount type=bind,source="$dataset",target=/data openvdb \
sh -c "python vrnens_data.py /data /data/converted --dataset $dataset_type" sh
