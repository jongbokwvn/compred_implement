num_gpus=$1
contextualize=$2
subset=$3
subreddit_name=$4
batch_size=$5

torchrun --nnodes 1 --nproc_per_node $num_gpus --master-port 29501 /home/wvnvwn/experiments/compred_private/raw_inferencer.py\
    --output_dir /home/wvnvwn/experiments/compred_private/inference_outputs/\
    --model_name microsoft/phi-3.5-mini-instruct\
    --dataset_name allenai/compred \
    --subset $subset\
    --subreddit_name $subreddit_name\
    --batch_size $batch_size\
    --contextualize $contextualize\