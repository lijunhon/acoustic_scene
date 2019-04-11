#!/bin/bash
# You need to modify this path
DATASET_DIR="/content/dcase2018_baseline/task1/datasets"

# You need to modify this path as your workspace
WORKSPACE="/content/drive/data"

DEV_SUBTASK_A_DIR="TUT-urban-acoustic-scenes-2018-development"
#LB_SUBTASK_A_DIR="TUT-urban-acoustic-scenes-2018-leaderboard"
EVAL_SUBTASK_A_DIR="TUT-urban-acoustic-scenes-2018-evaluation"

BACKEND="pytorch"	# "pytorch" | "keras"
HOLDOUT_FOLD=1
GPU_ID=0

############ Extract features ############
python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --data_type=development --workspace=$WORKSPACE
python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_B_DIR --data_type=development --workspace=$WORKSPACE
#python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$LB_SUBTASK_A_DIR --data_type=leaderboard --workspace=$WORKSPACE
#python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$LB_SUBTASK_B_DIR --data_type=leaderboard --workspace=$WORKSPACE
python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$EVAL_SUBTASK_A_DIR --data_type=evaluation --workspace=$WORKSPACE
python utils/features.py logmel --dataset_dir=$DATASET_DIR --subdir=$EVAL_SUBTASK_B_DIR --data_type=evaluation --workspace=$WORKSPACE

############ Development subtask A ############
# 模型训练
CUDA_VISIBLE_DEVICES=$GPU_ID python $BACKEND/main_$BACKEND.py train --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --workspace=$WORKSPACE --validate --holdout_fold=$HOLDOUT_FOLD --cuda

# 模型验证
CUDA_VISIBLE_DEVICES=$GPU_ID python $BACKEND/main_$BACKEND.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --workspace=$WORKSPACE --holdout_fold=$HOLDOUT_FOLD --iteration=5000 --cuda

############ Full train subtask A ############
# Train on full development data
CUDA_VISIBLE_DEVICES=$GPU_ID python $BACKEND/main_$BACKEND.py train --dataset_dir=$DATASET_DIR --subdir=$DEV_SUBTASK_A_DIR --workspace=$WORKSPACE --cuda

# Inference leaderboard data
#CUDA_VISIBLE_DEVICES=$GPU_ID python $BACKEND/main_$BACKEND.py inference_leaderboard_data --dataset_dir=$DATASET_DIR --dev_subdir=$DEV_SUBTASK_A_DIR --leaderboard_subdir=$LB_SUBTASK_A_DIR --workspace=$WORKSPACE --iteration=5000 --cuda

# Inference evaluation data
CUDA_VISIBLE_DEVICES=$GPU_ID python $BACKEND/main_$BACKEND.py inference_evaluation_data --dataset_dir=$DATASET_DIR --dev_subdir=$DEV_SUBTASK_A_DIR --eval_subdir=$EVAL_SUBTASK_A_DIR --workspace=$WORKSPACE --iteration=5000 --cuda

