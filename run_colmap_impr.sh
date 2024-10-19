# bash projects/neuralangelo/scripts2/f_colmap_known_dense.sh
Datapathraw=$1
Maskpathraw=$2
Datapath=$3

for arg in "$@"; do
    echo "------------- Input: $arg"
done

Densepath=$Datapath/dense
MaskDensepath=$Datapath/dense_mask
Sparsepath=$Datapath/sparse
Camerapath=$Datapath/camera

rm -rf $Datapath
mkdir $Datapath
colmap feature_extractor \
    --database_path $Datapath/database.db \
    --image_path $Datapathraw \
    --ImageReader.camera_model SIMPLE_RADIAL \
    --ImageReader.single_camera=true \
    --SiftExtraction.use_gpu=true \
    --SiftExtraction.num_threads=32 \
    --SiftExtraction.domain_size_pooling 1

colmap exhaustive_matcher \
    --database_path $Datapath/database.db \
    --SiftMatching.use_gpu=true \
    --ExhaustiveMatching.block_size 152

mkdir -p $Datapath/sparse

colmap mapper \
    --database_path $Datapath/database.db \
    --image_path $Datapathraw \
    --output_path $Datapath/sparse 

cp $Datapath/sparse/0/*.bin $Datapath/sparse/
for path in $Datapath/sparse/*/; do
    m=$(basename ${path})
    if [ ${m} != "0" ]; then
        colmap model_merger \
            --input_path1 $Datapath/sparse \
            --input_path2 $Datapath/sparse/${m} \
            --output_path $Datapath/sparse
        colmap bundle_adjuster \
            --input_path $Datapath/sparse \
            --output_path $Datapath/sparse
    fi
done

# ------------------------------------
mkdir $Densepath

colmap image_undistorter \
    --image_path $Datapathraw \
    --input_path $Datapath/sparse \
    --output_path $Densepath \
    --output_type=COLMAP

mkdir $MaskDensepath

colmap image_undistorter \
    --image_path $Maskpathraw \
    --input_path $Datapath/sparse \
    --output_path $MaskDensepath \
    --output_type=COLMAP

# ------------------------------------
colmap patch_match_stereo \
    --workspace_path $Densepath \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true

python transfer_mask.py \
    --dir $MaskDensepath/images \
    --dir_output $MaskDensepath/images_mask

colmap stereo_fusion \
    --workspace_path $Densepath \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path $Densepath/result.ply \
    --StereoFusion.mask_path $MaskDensepath/images_mask/

colmap stereo_fusion \
    --workspace_path $Densepath \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path $Densepath/result_unmask.ply 
    
mkdir $Camerapath

colmap model_converter \
    --input_path $Sparsepath \
    --output_path $Camerapath \
    --output_type TXT
