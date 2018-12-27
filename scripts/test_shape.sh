set -ex
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
PROJ_ROOT="$SCRIPTPATH/../"
GPU_IDS=${1}
DATASET_MODE=${2} # voxel or df
MODEL_NAME=${3} # model name



if [ ${USER} == 'junyanz' ]
then
  CHECKPOINTS_DIR=/data/vision/billf/scratch/junyanz/texture/ck_3d/}${MODEL_NAME}
else
  CHECKPOINTS_DIR=/data/vision/billf/scratch/ztzhang/texture/checkpoints_shape_df/}${MODEL_NAME}
fi

# training
for EPOCH in 50 100 150 200 250 300 350 400 450 500
  do
  echo ${EPOCH}
  python test_shape.py --gpu_ids ${GPU_IDS} \
      --checkpoints_dir ${CHECKPOINTS_DIR} \
      --model 'shape_gan' \
      --batch_size 16 \
      --n_shapes 32 \
      --th 0.012 \
      --epoch ${EPOCH}
  done
