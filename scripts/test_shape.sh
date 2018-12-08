set -ex
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
PROJ_ROOT="$SCRIPTPATH/../"
GPU_ID=${1}
DATASET_MODE=${2} # voxel or df
MODEL_NAME=${3} # model name



if [ ${USER} == 'junyanz' ]
then
  CHECKPOINTS_DIR=/data/vision/billf/scratch/junyanz/texture/ck_3d/
else
  CHECKPOINTS_DIR=/data/vision/billf/scratch/ztzhang/texture/checkpoints_shape_df/
fi

# training
for EPOCH in 50 100 150 200 250 300 350 400 450 500
  do
  echo ${EPOCH}
  CUDA_VISIBLE_DEVICES=${GPU_ID} python ./test_shape.py \
                      --checkpoints_dir ${CHECKPOINTS_DIR}${MODEL_NAME} \
                      --model 'shape_gan' \
                      --batch_size 16 \
                      --n_shapes 32 \
                      --th 0.012 \
                      --epoch ${EPOCH}
  done
