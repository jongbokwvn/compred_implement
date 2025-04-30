from dataclasses import dataclass, field
from typing import Optional
import torch

@dataclass
class TrainerArguments:
    #### Base Config ####
    output_dir: Optional[str] = field(default="trained_models/shp/sft", metadata={"help": "the directory where the trained model will be saved"})
    
    #### Model ####
    model_name: Optional[str] = field(default="microsoft/phi-3.5-mini-instruct", metadata={"help": "the model name"})
    
    #### LORA ####
    use_lora: Optional[bool] = field(default=False, metadata={"help": ""})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.1, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=64, metadata={"help": "the lora r parameter"})

    #### Low Resource: 4bit ####
    use_4bit: Optional[bool] = field(default=False, metadata={"help": ""})
    
    #### Dataset ####
    dataset_name: Optional[str] = field(default="allenai/compred", metadata={"help": "the dataset name"})
    subset: Optional[str] = field(default='politics', metadata={"help": "the subset to use"})
    subreddit_name: Optional[str] = field(default="none", metadata={"help": "the subreddit name to use"})
    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    contextualize: Optional[bool] = field(default=False, metadata={"help": ""})
    score_ratio_threshold: Optional[float] = field(default=2.0, metadata={"help": "the score ratio threshold"})
    num_examples_per_post: Optional[float] = field(default=5, metadata={"help": "the number of examples per post"})
    
    #### Data Loading ####
    size_valid_set: Optional[int] = field(default=4000, metadata={"help": "the size of the validation set"})
    streaming: Optional[bool] = field(default=False, metadata={"help": "whether to stream the dataset"})
    shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})
    torch_dtype: Optional[str] = field(default="bfloat16", metadata={"help": "the torch dtype"})
    seq_length: Optional[int] = field(default=1024, metadata={"help": "the sequence length"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})

    #### Logging ####
    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    #### Training ####
    max_steps: Optional[int] = field(default=500, metadata={"help": "the maximum number of sgd steps"})
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "the maximum number of sgd steps"})
    save_steps: Optional[int] = field(default=10000, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=500, metadata={"help": "the evaluation frequency"})
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "the per device train batch size"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "the per device eval batch size"})
    gradient_accumulation_steps: Optional[int] = field(default=128, metadata={"help": "the gradient accumulation steps"})
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    #### Optimizer ####
    learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "the learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_ratio: Optional[float] = field(default=0.03, metadata={"help": "warmup ratio"})
    weight_decay: Optional[float] = field(default=0.0, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})
    
@dataclass
class DPOTrainerArguments:
    #### Base Config ####
    output_dir: Optional[str] = field(default="trained_models/shp/dpo", metadata={"help": "the directory where the trained model will be saved"})
    
    #### Model ####
    model_name: Optional[str] = field(default="microsoft/phi-3.5-mini-instruct", metadata={"help": "the model name"})
    lora_checkpoint: Optional[str] = field(default=None, metadata={"help": "the lora checkpoint, checkpoint with num"})
    
    #### LORA ####
    use_lora: Optional[bool] = field(default=False, metadata={"help": ""})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.1, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=64, metadata={"help": "the lora r parameter"})

    #### Low Resource: 4bit ####
    use_4bit: Optional[bool] = field(default=True, metadata={"help": ""})
    
    #### Dataset ####
    dataset_name: Optional[str] = field(default="allenai/compred", metadata={"help": "the dataset name"})
    subset: Optional[str] = field(default='politics', metadata={"help": "the subset to use"})
    subreddit_name: Optional[str] = field(default="none", metadata={"help": "the subreddit name to use"})
    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    contextualize: Optional[bool] = field(default=True, metadata={"help": ""})
    score_ratio_threshold: Optional[float] = field(default=2.0, metadata={"help": "the score ratio threshold"})
    num_examples_per_post: Optional[float] = field(default=5, metadata={"help": "the number of examples per post"})
    
    #### Data Loading ####
    size_valid_set: Optional[int] = field(default=4000, metadata={"help": "the size of the validation set"})
    streaming: Optional[bool] = field(default=False, metadata={"help": "whether to stream the dataset"})
    shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})
    torch_dtype: Optional[str] = field(default="bfloat16", metadata={"help": "the torch dtype"})
    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    seq_length: Optional[int] = field(default=512, metadata={"help": "the sequence length"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})

    #### Logging ####
    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    #### Training ####
    max_steps: Optional[int] = field(default=1000, metadata={"help": "the maximum number of sgd steps"})
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "the maximum number of sgd steps"})
    save_steps: Optional[int] = field(default=10000, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency"})
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "the per device train batch size"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "the per device eval batch size"})
    gradient_accumulation_steps: Optional[int] = field(default=64, metadata={"help": "the gradient accumulation steps"})
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    #### Optimizer ####
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    learning_rate: Optional[float] = field(default=5e-7, metadata={"help": "the learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_ratio: Optional[float] = field(default=0.1, metadata={"help": "warmup ratio"})
    warmup_steps: Optional[int] = field(default=250, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.0, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})
    
    #### debugging arguments ####
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    ignore_bias_buffers: Optional[bool] = field(default=False, metadata={"help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"})

@dataclass
class AOTTrainerArguments:
    #### Base Config ####
    output_dir: Optional[str] = field(default="trained_models/shp/sft", metadata={"help": "the directory where the trained model will be saved"})
    
    #### Model ####
    model_name: Optional[str] = field(default="microsoft/phi-3.5-mini-instruct", metadata={"help": "the model name"})
    lora_checkpoint: Optional[str] = field(default=None, metadata={"help": "the lora checkpoint, checkpoint with num"})
    target_dist_path: Optional[str] = field(default=None, metadata={"help": "the target distribution path"})
    
    #### LORA ####
    use_lora: Optional[bool] = field(default=True, metadata={"help": ""})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.1, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=64, metadata={"help": "the lora r parameter"})
    
    #### Low Resource: 4bit ####
    use_4bit: Optional[bool] = field(default=True, metadata={"help": ""})
    
    #### Dataset ####
    dataset_name: Optional[str] = field(default="allenai/compred", metadata={"help": "the dataset name"})
    subset: Optional[str] = field(default='politics', metadata={"help": "the subset to use"})
    subreddit_name: Optional[str] = field(default="conservative", metadata={"help": "the subreddit name to use"})
    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    contextualize: Optional[bool] = field(default=False, metadata={"help": ""})
    score_ratio_threshold: Optional[float] = field(default=2.0, metadata={"help": "the score ratio threshold"})
    num_examples_per_post: Optional[int] = field(default=5, metadata={"help": "the number of examples per post"})
    
    #### Data Loading ####
    size_valid_set: Optional[int] = field(default=4000, metadata={"help": "the size of the validation set"})
    streaming: Optional[bool] = field(default=False, metadata={"help": "whether to stream the dataset"})
    shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})
    torch_dtype: Optional[torch.dtype] = field(default=torch.bfloat16, metadata={"help": "the torch dtype"})
    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    seq_length: Optional[int] = field(default=512, metadata={"help": "the sequence length"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})
    
    #### Logging ####
    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    
    #### Training #### 
    max_steps: Optional[int] = field(default=1000, metadata={"help": "the maximum number of sgd steps"})
    num_train_epochs: Optional[int] = field(default=5, metadata={"help": "the maximum number of sgd steps"})
    save_steps: Optional[int] = field(default=10000, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=500, metadata={"help": "the evaluation frequency"})
    per_device_train_batch_size: Optional[int] = field(default=1, metadata={"help": "the per device train batch size"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "the per device eval batch size"})
    gradient_accumulation_steps: Optional[int] = field(default=128, metadata={"help": "the gradient accumulation steps"})
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )
    #### Optimizer ####
    lambda_align: Optional[float] = field(default=1.0, metadata={"help": "the lambda align parameter"})
    learning_rate: Optional[float] = field(default=1e-5, metadata={"help": "the learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_ratio: Optional[float] = field(default=0.1, metadata={"help": "warmup ratio"})
    weight_decay: Optional[float] = field(default=0.0, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    #### Debugging Arguments ####

@dataclass
class InferencerArguments:
    #### Base Config ####
    output_dir: Optional[str] = field(default=None, metadata={"help": "the reward model name"})
    
    #### Model ####
    model_name: Optional[str] = field(default="microsoft/phi-3.5-mini-instruct", metadata={"help": "the model name or final_merged_checkpoint"})
    lora_checkpoint: Optional[str] = field(default=None, metadata={"help": "the lora checkpoint, checkpoint with num"})
    
    #### LORA ####
    use_lora: Optional[bool] = field(default=True, metadata={"help": "use lora for inference"})
    lora_r: Optional[int] = field(default=64, metadata={"help": "the lora r parameter"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.1, metadata={"help": "the lora dropout parameter"})
    
    #### Low Resource: 4bit ####
    use_4bit: Optional[bool] = field(default=False, metadata={"help": ""})
    
    #### Dataset ####
    dataset_name: Optional[str] = field(default="allenai/compred", metadata={"help": "the dataset name"})
    subset: str = field(default='politics', metadata={"help": "which subset to use"})
    subreddit_name: str = field(default='askaliberal', metadata={"help": "which subreddit to use"})
    contextualize: Optional[bool] = field(default=False, metadata={"help": "whether to add subreddit context"})
    
    #### Data Loading ####
    torch_dtype: Optional[str] = field(default="bfloat16", metadata={"help": "the torch dtype"})
    seq_length: Optional[int] = field(default=512, metadata={"help": "the sequence length"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})
    
    #### Inference ####
    batch_size: Optional[int] = field(default=1, metadata={"help": "decoding batch size"})
    randomize_context: Optional[bool] = field(default=False, metadata={"help": "add a random context to the model instead of the provided context"})

@dataclass
class ComPOInferencerArguments:
    #### Base Config ####
    output_dir: Optional[str] = field(default=None, metadata={"help": "the reward model name"})
    
    #### Model ####
    model_name: Optional[str] = field(default="microsoft/phi-3.5-mini-instruct", metadata={"help": "the model name or final_merged_checkpoint"})
    lora_checkpoint: Optional[str] = field(default=None, metadata={"help": "the lora checkpoint, checkpoint with num"})
    
    #### LORA ####
    use_lora: Optional[bool] = field(default=True, metadata={"help": "use lora for inference"})
    lora_r: Optional[int] = field(default=64, metadata={"help": "the lora r parameter"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.1, metadata={"help": "the lora dropout parameter"})
    
    #### Low Resource: 4bit ####
    use_4bit: Optional[bool] = field(default=False, metadata={"help": ""})
    
    #### Dataset ####
    dataset_name: Optional[str] = field(default="allenai/compred", metadata={"help": "the dataset name"})
    subset: str = field(default='politics', metadata={"help": "which subset to use"})
    subreddit_name: str = field(default='askaliberal', metadata={"help": "which subreddit to use"})
    contextualize: Optional[bool] = field(default=False, metadata={"help": "whether to add subreddit context"})
    
    #### Data Loading ####
    torch_dtype: Optional[str] = field(default="bfloat16", metadata={"help": "the torch dtype"})
    seq_length: Optional[int] = field(default=512, metadata={"help": "the sequence length"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})
    
    #### Inference ####
    batch_size: Optional[int] = field(default=1, metadata={"help": "decoding batch size"})
    randomize_context: Optional[bool] = field(default=False, metadata={"help": "add a random context to the model instead of the provided context"})
    
@dataclass
class AOTInferencerArguments:
    #### Base Config ####
    output_dir: Optional[str] = field(default=None, metadata={"help": "the reward model name"})
    
    #### Model ####
    model_name: Optional[str] = field(default="microsoft/phi-3.5-mini-instruct", metadata={"help": "the model name or final_merged_checkpoint"})
    model_state_path: Optional[str] = field(default="/home/wvnvwn/experiments/compred_private/models/aot-c/politics/none/microsoft-phi-3.5-mini-instruct/aot_cons_dist_none.pt", metadata={"help": "the path to the model state"})
    
    #### LORA ####
    use_lora: Optional[bool] = field(default=True, metadata={"help": "use lora for inference"})
    lora_r: Optional[int] = field(default=64, metadata={"help": "the lora r parameter"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.1, metadata={"help": "the lora dropout parameter"})
    
    #### Low Resource: 4bit ####
    use_4bit: Optional[bool] = field(default=False, metadata={"help": ""})
    
    #### Dataset ####
    dataset_name: Optional[str] = field(default="allenai/compred", metadata={"help": "the dataset name"})
    subset: str = field(default='politics', metadata={"help": "which subset to use"})
    subreddit_name: str = field(default='askaliberal', metadata={"help": "which subreddit to use"})
    contextualize: Optional[bool] = field(default=False, metadata={"help": "whether to add subreddit context"})
    
    #### Data Loading ####
    torch_dtype: Optional[str] = field(default="bfloat16", metadata={"help": "the torch dtype"})
    seq_length: Optional[int] = field(default=512, metadata={"help": "the sequence length"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})
    
    #### Inference ####
    batch_size: Optional[int] = field(default=1, metadata={"help": "decoding batch size"})
    randomize_context: Optional[bool] = field(default=False, metadata={"help": "add a random context to the model instead of the provided context"})
    
@dataclass
class StanceClassifierArguments:
    #### Base Config ####
    input_dir: Optional[str] = field(default=None, metadata={"help": "the input directory"})
    input_file_name: Optional[str] = field(default=None, metadata={"help": "the input file name"})
    output_dir: Optional[str] = field(default=None, metadata={"help": "the reward model name"})
    
    ### Model ###
    model_name: Optional[str] = field(default="microsoft/phi-3.5-mini-instruct", metadata={"help": "the model name"})
    device: Optional[int] = field(default=-1, metadata={"help": "the device to use"})