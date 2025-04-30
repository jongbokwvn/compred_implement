num_gpus=$1
model_path=$2
lora_checkpoint=$3
subset=$4
subreddit_name=$5
contextualize=$6

torchrun --nnodes 1 --nproc_per_node $num_gpus --master_port 29502 /home/wvnvwn/experiments/compred_private/sft_inferencer.py\
    --output_dir /home/wvnvwn/experiments/compred_private/inference_outputs/\
    --model_name $model_path\
    --lora_checkpoint $lora_checkpoint\
    --dataset_name allenai/compred \
    --subset $subset\
    --use_4bit True\
    --subreddit_name $subreddit_name\
    --batch_size 1\
    --contextualize $contextualize\