set -ex
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
GPU_IDS=${1}
DATASET_MODE=${2} # df | voxel
MODEL_NAME=${3} # model's full name, not only the class name

CHECKPOINTS_DIR=${SCRIPTPATH}/ck_shape/${MODEL_NAME}

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
