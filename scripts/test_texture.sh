

# misc
set -ex
GPU_ID=${1}
CLASS=${2}
DATASET=${3}

# models
DISPLAY_ID=$((GPU_ID*10+1))
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
ROOT_DIR=${SCRIPTPATH}/..


MODEL2D_DIR=${ROOT_DIR}/final_models/models_2D/${CLASS}_${DATASET}/latest
MODEL3D_DIR=${ROOT_DIR}/final_models/models_3D/${CLASS}_${DATASET}
RESULTS_DIR=${ROOT_DIR}/results/test_texture_${CLASS}_${DATASET}/


NUM_SHAPES=10   # number of shapes duirng test
NUM_SAMPLES=5  # number of samples per shape

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./test.py \
  --results_dir ${RESULTS_DIR} \
  --model2D_dir ${MODEL2D_DIR} \
  --model3D_dir ${MODEL3D_DIR} \
  --class_3d ${CLASS} \
  --phase 'val' \
  --dataset_mode 'image_and_'${DATASET} \
  --resize_or_crop 'crop_real_im' \
  --model 'test'  \
  --norm 'inst' \
  --verbose \
  --load_size 128 --fine_size 128 \
  --df_th 0.90 \
  --seed 10 \
  --n_shapes ${NUM_SHAPES} \
  --n_views ${NUM_SAMPLES} \
  --show_input \
  --reset_shape \
  --reset_texture \
  --suffix ${CLASS}_${DATASET}_t{real_texture} ${4}
