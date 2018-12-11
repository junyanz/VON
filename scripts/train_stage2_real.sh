set -ex

GPU_ID=${1}
CLASS=${2}
DATASET=${3}
DF_TH=0.9
DATE=`date +%Y-%m-%d`
DISPLAY_ID=$((GPU_ID*10+1))

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
ROOT_DIR=${SCRIPTPATH}/..
MODEL2D_DIR=${ROOT_DIR}/final_models/models_2D/${CLASS}_${DATASET}/latest
MODEL3D_DIR=${ROOT_DIR}/final_models/models_3D/${CLASS}_${DATASET}
CHECKPOINTS_DIR=${ROOT_DIR}/checkpoints/stage2_real/${CLASS}_${DATASET}/${DATE}/



# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train.py \
  --display_id 1001 \
  --dataset_mode 'image_and_'${DATASET} \
  --resize_or_crop 'crop_real_im' \
  --model 'stage2_real' \
  --checkpoints_dir ${CHECKPOINTS_DIR} \
  --class_3d ${CLASS} \
  --random_shift \
  --color_jitter \
  --norm 'inst' \
  --verbose \
  --df_th ${DF_TH} \
  --batch_size 12 --num_threads 6 \
  --gan_mode 'lsgan' \
  --suffix {class_3d}_${DATASET}_d2 \
