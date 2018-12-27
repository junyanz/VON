# misc
set -ex
GPU_IDS=${1}   # 0
CLASS=${2}    # car | chair
DATASET=${3}  # df | voxel
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
ROOT_DIR=${SCRIPTPATH}/..
# models
MODEL2D_DIR=${ROOT_DIR}/final_models/models_2D/${CLASS}_${DATASET}/latest
MODEL3D_DIR=${ROOT_DIR}/final_models/models_3D/${CLASS}_${DATASET}
RESULTS_DIR=${ROOT_DIR}/results/fig_${CLASS}_${DATASET}/

NUM_SHAPES=20   # number of shapes duirng test
NUM_SAMPLES=5  # number of samples per shape

# command
python test.py --gpu_ids ${GPU_IDS} \
  --results_dir ${RESULTS_DIR} \
  --model2D_dir ${MODEL2D_DIR} \
  --model3D_dir ${MODEL3D_DIR} \
  --class_3d ${CLASS} \
  --phase 'val' \
  --dataset_mode 'image_and_'${DATASET} \
  --model 'test'  \
  --n_shapes ${NUM_SHAPES} \
  --n_views ${NUM_SAMPLES} \
  --reset_texture \
  --reset_shape \
  --suffix ${CLASS}_${DATASET}
