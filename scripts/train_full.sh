set -ex

GPU_ID=${1}
CLASS=${2}
DATASET=${3}
DATE=`date +%Y-%m-%d`


# training
DISPLAY_ID=$((GPU_ID*10+1))
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
ROOT_DIR=${SCRIPTPATH}/..
MODEL2D_DIR=${ROOT_DIR}/final_models/models_2D/${CLASS}_${DATASET}/latest
MODEL3D_DIR=${ROOT_DIR}/final_models/models_3D/${CLASS}_${DATASET}
CHECKPOINTS_DIR=${ROOT_DIR}/checkpoints/full/${CLASS}_${DATASET}/${DATE}/

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train.py \
  --display_id ${DISPLAY_ID} \
  --dataset_mode image_and_${DATASET} \
  --model 'full' \
  --class_3d ${CLASS} \
  --checkpoints_dir ${CHECKPOINTS_DIR} \
  --model2D_dir ${MODEL2D_DIR} \
  --model3D_dir ${MODEL3D_DIR} \
  --lambda_mask 2.5 --df_th 0.90 \
  --norm 'inst' --netD 'multi' --num_Ds 2 \
  --random_shift \
  --color_jitter \
  --lambda_GAN_3D 0.05 \
  --suffix {model}_{class_3d}_${DATASET}
