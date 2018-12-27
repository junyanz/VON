set -ex

GPU_IDS=${1}
CLASS=${2}  # car or chair
DATASET=${3} # df or voxel
DATE=`date +%Y-%m-%d`
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
ROOT_DIR=${SCRIPTPATH}/..
CHECKPOINTS_DIR=${ROOT_DIR}/checkpoints/shape/${CLASS}_${DATASET}/${DATE}/

# training
python train.py --gpu_ids ${GPU_IDS} \
                  --display_id 1000 \
                  --dataset_mode ${DATASET} \
                  --model 'shape_gan' \
                  --class_3d ${CLASS} \
                  --checkpoints_dir ${CHECKPOINTS_DIR} \
                  --niter 250 --niter_decay 250 \
                  --batch_size 8 \
                  --save_epoch_freq 10 \
                  --suffix {class_3d}_{model}_{dataset_mode} \
