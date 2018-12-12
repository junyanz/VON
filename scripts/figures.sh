# misc
set -ex
GPU_ID=${1}   # 0
CLASS=${2}    # car, chair
DATASET=${3}  # df, voxel
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
ROOT_DIR=${SCRIPTPATH}/..
# models
DISPLAY_ID=$((GPU_ID*10+1))
MODEL2D_DIR=${ROOT_DIR}/final_models/models_2D/${CLASS}_${DATASET}/latest
MODEL3D_DIR=${ROOT_DIR}/final_models/models_3D/${CLASS}_${DATASET}
RESULTS_DIR=${ROOT_DIR}/results/fig_${CLASS}_${DATASET}/

NUM_SHAPES=20   # number of shapes duirng test
NUM_SAMPLES=8  # number of samples per shape

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
  --load_size 128 --fine_size 128 \
  --df_th 0.90 \
  --n_shapes ${NUM_SHAPES} \
  --n_views ${NUM_SAMPLES} \
  --real_texture \
  --reset_texture \
  --show_input \
  --suffix ${CLASS}_${DATASET} \
  --batch_size 1 \
  --use_df \
