num_gpus=$1
lora_checkpoint=$2
subset=$3
subreddit_name=$4
contextualize=$5

torchrun --nnodes 1 --nproc_per_node $num_gpus --master_port 29501 /home/wvnvwn/experiments/compred_private/dpo_trainer.py\
    --output_dir /home/wvnvwn/experiments/compred_private/models/dpo-nc/\
    --model_name microsoft/phi-3.5-mini-instruct \
    --lora_checkpoint $lora_checkpoint \
    --use_lora True \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lora_r 64 \
    --use_4bit True \
    --dataset_name allenai/compred \
    --subset $subset\
    --subreddit_name $subreddit_name\
    --contextualize $contextualize\
    --no_gradient_checkpointing\
    --per_device_train_batch_size 1\
    --per_device_eval_batch_size 1\
    --learning_rate 5e-7 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --weight_decay 0.0