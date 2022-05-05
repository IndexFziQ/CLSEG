PER_GPU_BATCH_SIZE=1
BATCH_SIZE=32
MAX_LEN=100
GC_STEP=32
GPU_NUM=1

LR_RATE=1 # lr_iter=(1) #  (1 0.5 2)
EPOCH_N=3 # epoch_iter=(70)
SEED=42 # seed_iter=(42 100 512 1024 2019)

TASK_NAME='lmft'
NET_NAME='lm'
M_NAME='gpt2_rocmedium'

BASE_DIR=/home/yuqiang.xyq/unified-story-level-pretraining/src
DATA_DIR=/data/xieyuqiang/story_comprehension/data
MODEL_PATH=/data/pretrained_models/pytorch-gpt2-medium
TK_PATH=/data/pretrained_models/pytorch-gpt2-medium

OUTPUT_BASE=/data/xieyuqiang/story_comprehension/checkpoints_ft_with_roc

EXP_PRE="seed${SEED}_cgn1"

OUTPUT_DIR=${OUTPUT_BASE}/${TASK_NAME}_${NET_NAME}_${M_NAME}_gn${GPU_NUM}_bp${PER_GPU_BATCH_SIZE}_gc${GC_STEP}_lr${LR_RATE}_l${MAX_LEN}_e${EPOCH_N}_${EXP_PRE}
mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=0 \
python -u run_lm_finetune.lm.roc.py \
--overwrite_output_dir \
--overwrite_cache \
--task_name ${TASK_NAME} \
--data_dir ${DATA_DIR} \
--train_file ${DATA_DIR}/roc/roc_train.csv \
--model_type gpt2 \
--network_name ${NET_NAME} \
--model_name_or_path ${MODEL_PATH} \
--tokenizer_name_or_path ${TK_PATH} \
--output_dir ${OUTPUT_DIR} \
--log_file ${BASE_DIR}/log.${TASK_NAME}.${NET_NAME}.${M_NAME}.gn${GPU_NUM}.bp${PER_GPU_BATCH_SIZE}.gc${GC_STEP}.lr${LR_RATE}.l${MAX_LEN}.e${EPOCH_N}.${EXP_PRE}.out \
--tfboard_log_dir ${OUTPUT_DIR}/tfboard.event.out \
--block_size ${MAX_LEN} \
--do_train \
--do_lower_case \
--train_batch_size ${BATCH_SIZE} \
--per_gpu_train_batch_size ${PER_GPU_BATCH_SIZE} \
--per_gpu_eval_batch_size 2 \
--num_train_epochs ${EPOCH_N} \
--max_steps -1 \
--learning_rate ${LR_RATE}e-5 \
--gradient_accumulation_steps ${GC_STEP} \
--max_grad_norm 1.0 \
--adam_epsilon 1e-8 \
--weight_decay 0.0 \
--seed ${SEED} \
--save_steps 5000
