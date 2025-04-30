num_gpus=$1
lora_checkpoint=$2
subset=$3
subreddit_name=$4
contextualize=$5

export CUDA_VISIBLE_DEVICES="1, 5"

torchrun --nnodes 1 --nproc_per_node $num_gpus --master_port 29501 /home/wvnvwn/experiments/compred_private/aot_inferencer.py\
    --output_dir /home/wvnvwn/experiments/compred_private/inference_outputs/\
    --model_name microsoft/phi-3.5-mini-instruct \
    --model_state_path $lora_checkpoint\
    --dataset_name allenai/compred \
    --subset $subset\
    --subreddit_name $subreddit_name\
    --batch_size 1\
    --contextualize $contextualize\