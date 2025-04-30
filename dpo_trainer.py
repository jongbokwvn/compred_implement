# 0. imports
import os
import gc
import json
import wandb
import torch
import torch.distributed as dist
import logging
import warnings
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser
)
from transformers.trainer_callback import ProgressCallback
from trl import DPOTrainer, DPOConfig
from trl.trainer.dpo_trainer import DataCollatorForPreference
from safetensors.torch import load_file
from arguments import DPOTrainerArguments
from utils import (
    DDPManager, create_dataset, FormattingFunction,
    subsample, get_paired_dataset, remove_module_prefix,
    rename_lora_keys
)
from constants import HF_TOKEN
from huggingface_hub.commands.user import login; login(token=HF_TOKEN)


def main():
    # parsing arguments
    parser = HfArgumentParser(DPOTrainerArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    # Setting NCCL timeout
    os.environ["NCCL_TIMEOUT"] = "7200"  # 1시간으로 설정
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"  # 새로운 환경변수 사용
    os.environ["NCCL_BLOCKING_WAIT"] = "1"  # 블로킹 대기 모드 활성화
    # os.environ["NCCL_P2P_DISABLE"] = "1"  # P2P 비활성화
    # os.environ["NCCL_IB_DISABLE"] = "1"  # IB 비활성화
    
    # initialize DDP
    ddp_manager = DDPManager()
    rank, world_size, local_rank = ddp_manager.rank, ddp_manager.world_size, ddp_manager.local_rank
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    print("########################## Device is set #########################")
    
    # setting output directory
    path_name = os.path.join(
        script_args.output_dir, script_args.subset, script_args.subreddit_name, script_args.model_name.replace("/", "-")
    ) if script_args.contextualize \
        else os.path.join(
            script_args.output_dir, script_args.subset, script_args.subreddit_name, script_args.model_name.replace("/", "-")
        )
        
    script_args.output_dir = path_name
    os.makedirs(script_args.output_dir, exist_ok=True)
    
    # Excusing unncessary warnings 
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.getLogger('trl').setLevel(logging.ERROR)
    warnings.filterwarnings('ignore', category=UserWarning, module='trl.trainer.utils')
    
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name,
        trust_remote_code=True,
        token=HF_TOKEN,
        padding=True,
        padding_side="left",
        truncation=True,
        max_length=script_args.seq_length,
        return_tensors="pt"
    )
    if getattr(tokenizer, "pad_token", None) is None:
        num_added_tokens = tokenizer.add_special_tokens({
            "pad_token": "<pad>",
        })
        assert num_added_tokens in [0, 1], "The Tokenizer should only add one special token - the pad_token, or no tokens if pad token present."
    print("######################### Tokenizer is loaded #########################")

    '''
    model: 학습 대상이 되는 모델 (학습 과정에서 파라미터 변화)
    model_ref: 비교 대상이 되는 모델 (학습 과정에서 파라미터 불변)
    
    model과 model_ref는 동일한 모델 -> 훈련 과정에서 model만 변화하며 달라지게 됨
    '''
    #### for base_model ####
    # for low resource mode
    if script_args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            script_args.model_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
            token=HF_TOKEN,
            torch_dtype=script_args.torch_dtype
        )
        base_model.config.use_cache = False
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            script_args.model_name,
            trust_remote_code=True,
            token=HF_TOKEN,
            torch_dtype=script_args.torch_dtype
        )
        base_model.config.use_cache = False
    
    # load LoRA check
    if script_args.lora_checkpoint:
        checkpoint_path = script_args.lora_checkpoint
        # final_merged_checkpoint
        if os.path.basename(checkpoint_path) == "final_merged_checkpoint":
            print(f"Loading merged checkpoint from {checkpoint_path}")
            base_model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                trust_remote_code=True,
                torch_dtype=script_args.torch_dtype,
                is_trainable=True
            )
        # LORA checkpoint
        else:
            print(f"Loading LORA checkpoint from {checkpoint_path}")
            if os.path.exists(os.path.join(checkpoint_path, "adapter_model.safetensors")):
                # PEFT/LORA 설정 로드
                with open(os.path.join(checkpoint_path, "adapter_config.json"), 'r') as f:
                    adapter_config = json.load(f)
                
                lora_target_modules = ['qkv_proj', 'o_proj', 'gate_up_proj', 'down_proj', 'embed_tokens']
                
                lora_config = LoraConfig(
                    r=adapter_config.get('r', script_args.lora_r),
                    lora_alpha=adapter_config.get('lora_alpha', script_args.lora_alpha),
                    lora_dropout=adapter_config.get('lora_dropout', script_args.lora_dropout),
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=lora_target_modules
                )
                
                # initialize PEFT model
                base_model = get_peft_model(base_model, lora_config)
                
                # safetensors에서 가중치 로드
                state_dict = load_file(os.path.join(checkpoint_path, "adapter_model.safetensors"))
                state_dict = remove_module_prefix(state_dict)
                state_dict = rename_lora_keys(state_dict)
                
                print(f"#### Loaded state_dict from {checkpoint_path} with {len(state_dict.keys())} keys.")
                some_keys = list(state_dict.keys())[:10]  # 앞 몇 개만 출력
                print(f"#### Some of the keys: {some_keys} ...")

                _, unexpected_keys = base_model.load_state_dict(state_dict, strict=False)
                print(f"#### Unexpected keys: {unexpected_keys}")
                
            else:
                raise ValueError(f"Cannot find adapter_model.safetensors in {checkpoint_path}")
    print("######################### Model is loaded #########################")
    
    #### for model_ref ####
    # for low resource mode
    if script_args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model_ref = AutoModelForCausalLM.from_pretrained(
            script_args.model_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
            token=HF_TOKEN,
            torch_dtype=script_args.torch_dtype
        )
        base_model_ref.config.use_cache = False
    else:
        base_model_ref = AutoModelForCausalLM.from_pretrained(
            script_args.model_name,
            trust_remote_code=True,
            token=HF_TOKEN,
            torch_dtype=script_args.torch_dtype
        )
        base_model_ref.config.use_cache = False
    
    # load LoRA check
    if script_args.lora_checkpoint:
        checkpoint_path = script_args.lora_checkpoint
        # final_merged_checkpoint
        if os.path.basename(checkpoint_path) == "final_merged_checkpoint":
            print(f"Loading merged checkpoint from {checkpoint_path}")
            base_model_ref = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                trust_remote_code=True,
                torch_dtype=script_args.torch_dtype,
            )
        # LORA checkpoint
        else:
            print(f"Loading LORA checkpoint from {checkpoint_path}")
            if os.path.exists(os.path.join(checkpoint_path, "adapter_model.safetensors")):
                # PEFT/LORA 설정 로드
                with open(os.path.join(checkpoint_path, "adapter_config.json"), 'r') as f:
                    adapter_config = json.load(f)
                
                lora_target_modules = ['qkv_proj', 'o_proj', 'gate_up_proj', 'down_proj', 'embed_tokens']
                
                lora_config = LoraConfig(
                    r=adapter_config.get('r', script_args.lora_r),
                    lora_alpha=adapter_config.get('lora_alpha', script_args.lora_alpha),
                    lora_dropout=adapter_config.get('lora_dropout', script_args.lora_dropout),
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=lora_target_modules
                )
                
                # initialize PEFT model
                base_model_ref = get_peft_model(base_model_ref, lora_config)
                
                # safetensors에서 가중치 로드
                state_dict = load_file(os.path.join(checkpoint_path, "adapter_model.safetensors"))
                state_dict = remove_module_prefix(state_dict)
                state_dict = rename_lora_keys(state_dict)
                
                print(f"#### Loaded state_dict from {checkpoint_path} with {len(state_dict.keys())} keys.")
                some_keys = list(state_dict.keys())[:10]  # 앞 몇 개만 출력
                print(f"#### Some of the keys: {some_keys} ...")

                _, unexpected_keys = base_model_ref.load_state_dict(state_dict, strict=False)
                print(f"#### Unexpected keys: {unexpected_keys}")
                
            else:
                raise ValueError(f"Cannot find adapter_model.safetensors in {checkpoint_path}")
    
    print("######################### Model_ref is loaded #########################")
    
    '''
    basd_model_ref는 ddp로 wrapping 되지 아니함
        - base_model의 훈련 과정에서 forward pass에서만 사용하기 때문
        - parameter update 및 gradient 계산 없기 때문
    '''
    if world_size > 1:
        base_model = torch.nn.parallel.DistributedDataParallel(
            base_model.to(device),
            device_ids=[local_rank],
            output_device=local_rank
        )
        base_model.config = base_model.module.config
        base_model.warnings_issued = getattr(base_model.module, 'warnings_issued', {})
        print(f"#### [Rank {local_rank}] Model is wrapped with DDP")
    else:
        base_model.to(device)
        
    base_model.train()
    base_model_ref.eval()
    print("######################### Model is moved to device #########################")
    
    '''
    sft trainer와 동일한 갯수의 데이터로 훈련을 위해
    길이 기반의 filter 조건은 사용하지 않음
    '''
    # load train dataset
    train_dataset = get_paired_dataset(
        script_args=script_args,
        data_path=script_args.dataset_name,
        split="train_pref",
        subset=script_args.subset,
        sanity_check=script_args.sanity_check,
        tokenizer=tokenizer
    )
    # train_dataset = train_dataset.filter(
    #     lambda x: len(x['prompt']) + len(x['chosen']) <= script_args.seq_length
    #     and len(x['prompt']) + len(x['rejected']) <= script_args.seq_length
    # )
    # load eval dataset
    eval_dataset = get_paired_dataset(
        script_args=script_args,
        data_path=script_args.dataset_name,
        split="validation_pref",
        subset=script_args.subset,
        sanity_check=True,
        tokenizer=tokenizer
    )
    # eval_dataset = eval_dataset.filter(
    #     lambda x: len(x['prompt']) + len(x['chosen']) <= script_args.seq_length
    #     and len(x['prompt']) + len(x['rejected']) <= script_args.seq_length
    # )
    
    print(f"#### train_dataset: {len(train_dataset)}")
    print(f"#### eval_dataset: {len(eval_dataset)}")
    print("######################### Successfully created datasets #########################")
    
    training_args = DPOConfig(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        # max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        save_total_limit = 1,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        eval_strategy="steps",
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.log_with,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name="compo",
        ddp_find_unused_parameters=False,
        beta = 0.1,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.seq_length,
        force_use_ref_model=True,
    )
    print("######################### TrainingArguments are set #########################")
    
    callbacks = [ProgressCallback()]
    data_collator = DataCollatorForPreference(pad_token_id=tokenizer.pad_token_id)
    
    if script_args.use_lora:
        dpo_trainer = DPOTrainer(
            base_model,
            base_model_ref,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks
        )
    else:
        dpo_trainer = DPOTrainer(
            base_model,
            base_model_ref,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks
        )

    print(f"#### dpo_trainer: {dpo_trainer}")
    print(f"#### dpo_trainer.train_dataset: {dpo_trainer.train_dataset}")
    print("########################## DPO Trainer is set #########################")
    
    dpo_trainer.train()
    print("########################## Training is completed #########################")

    # to force the model to be saved only once in main process
    if not dist.is_initialized() or dist.get_rank() == 0:
        model_to_save = dpo_trainer.model.module if hasattr(dpo_trainer.model, "module") else dpo_trainer.model
        model_to_save.save_pretrained(script_args.output_dir)
        print(f"##### Model is saved to {script_args.output_dir}")
    
    if script_args.use_lora:
        output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
        model_to_save = dpo_trainer.model.module if hasattr(dpo_trainer.model, 'module') else dpo_trainer.model
        # save the model: LoRA
        model_to_save.save_pretrained(output_dir)
        print(f"#### Model to save: {model_to_save}")

        del dpo_trainer
        del base_model
        gc.collect()
        torch.cuda.empty_cache()

        model = AutoPeftModelForCausalLM.from_pretrained(
            output_dir,
            torch_dtype=script_args.torch_dtype,
            ignore_mismatched_sizes=True,
        )   
        
        # merge LoRA weights with orginal model
        model = model.merge_and_unload()

        output_merged_dir = os.path.join(script_args.output_dir, "final_merged_checkpoint")
        # save the merged model
        model.save_pretrained(output_merged_dir, safe_serialization=True)
        print(f"#### Final Merged Model is saved to {output_merged_dir}")

    ddp_manager.cleanup()
        
if __name__ == "__main__":
    # GPU Setting
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
    main()