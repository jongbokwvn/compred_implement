num_gpus=$1
input_dir=$2
input_file_name=$3

python /home/wvnvwn/experiments/compred_private/stance_classifier.py\
    --model_name microsoft/phi-3.5-mini-instruct\
    --device $num_gpus\
    --input_dir $input_dir\
    --output_dir /home/wvnvwn/experiments/compred_private/stance_classifier/\
    --input_file_name $input_file_name