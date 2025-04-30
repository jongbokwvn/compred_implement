num_gpus=$1
subset=$2
subreddit_name=$3
contextualize=$4
num_workers=$5

torchrun --nnodes 1 --nproc_per_node $num_gpus --master_port 29502 /home/wvnvwn/experiments/compred_private/sft_trainer.py\
    --output_dir /home/wvnvwn/experiments/compred_private/models/\
    --model_name microsoft/phi-3.5-mini-instruct \
    --use_lora True \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lora_r 64 \
    --use_4bit False \
    --dataset_name allenai/compred \
    --subset $subset\
    --subreddit_name $subreddit_name\
    --contextualize $contextualize\
    --num_workers $num_workers \
    --no_gradient_checkpointing\
    --per_device_train_batch_size 1\
    --per_device_eval_batch_size 1\
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.0