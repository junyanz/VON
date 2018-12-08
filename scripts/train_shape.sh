set -ex

GPU_ID=${1}
CLASS=${2}  # car or chair
DATASET=${3} # df or voxel
DATE=`date +%Y-%m-%d`
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
ROOT_DIR=${SCRIPTPATH}/..
CHECKPOINTS_DIR=${ROOT_DIR}/checkpoints/shape/${CLASS}_${DATASET}/${DATE}/

# training
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train.py \
                    --display_id 1000 \
                    --dataset_mode ${DATASET} \
                    --model 'shape_gan' \
                    --class_3d ${CLASS} \
                    --D_norm_3D none \
                    --checkpoints_dir ${CHECKPOINTS_DIR} \
                    --niter 250 --niter_decay 250 \
                    --batch_size 8 \
                    --save_epoch_freq 10 \
                    --suffix {class_3d}_{model}_{dataset_mode}_D_{D_norm_3D} \
                    --display_port 6543
