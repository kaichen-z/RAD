Datapathraw=$1
Maskpathraw=$2
Campath=$3
Datapath=$4

Densepath=$Datapath/dense
MaskDensepath=$Datapath/dense_mask

# Create directories if they don't exist
mkdir -p $Densepath
mkdir -p $MaskDensepath

# Undistort the image
colmap image_undistorter \
    --image_path $Datapathraw \
    --input_path $Campath \
    --output_path $Densepath \
    --output_type=COLMAP

# Undistort the mask
colmap image_undistorter \
    --image_path $Maskpathraw \
    --input_path $Campath \
    --output_path $MaskDensepath \
    --output_type=COLMAP

Sparsepath=$Densepath/sparse
Camerapath=$Datapath/camera
mkdir $Camerapath
colmap model_converter \
    --input_path $Sparsepath \
    --output_path $Camerapath \
    --output_type TXT