import os
import gc
import torch
import torch.distributed as dist
import logging
import warnings
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser
)
from transformers.trainer_callback import ProgressCallback
from trl import SFTTrainer, SFTConfig
from trl.trainer.utils import DataCollatorForCompletionOnlyLM

from arguments import TrainerArguments
from utils import DDPManager, create_dataset, FormattingFunction
from constants import HF_TOKEN
from huggingface_hub.commands.user import login; login(token=HF_TOKEN)


def main():
    # parsing arguments
    parser = HfArgumentParser(TrainerArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    # Setting NCCL timeout
    os.environ["NCCL_TIMEOUT"] = "3600"  # 1시간으로 설정
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"  # 새로운 환경변수 사용
    os.environ["NCCL_BLOCKING_WAIT"] = "1"  # 블로킹 대기 모드 활성화
    os.environ["NCCL_P2P_DISABLE"] = "1"  # P2P 비활성화
    os.environ["NCCL_IB_DISABLE"] = "1"  # IB 비활성화
    
    # initialize DDP
    ddp_manager = DDPManager()
    rank, world_size, local_rank = ddp_manager.rank, ddp_manager.world_size, ddp_manager.local_rank
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    print("######################### Device is set #########################")
    
    # setting output directory
    path_name = os.path.join(
        script_args.output_dir, "sft-c", script_args.subset, script_args.subreddit_name, script_args.model_name.replace("/", "-")
    ) if script_args.contextualize \
        else os.path.join(
            script_args.output_dir, "sft-nc", script_args.subset, script_args.subreddit_name, script_args.model_name.replace("/", "-")
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

    # only valid for phi-3.5-mini-instruct
    lora_target_modules = ['qkv_proj', 'o_proj', 'gate_up_proj', 'down_proj', 'embed_tokens']
    
    if script_args.use_lora:
        peft_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        base_model = get_peft_model(base_model, peft_config)
    else:
        peft_config = None
    print("######################### Model is loaded #########################")
    
    # activate DDP iff world_size >1
    if world_size > 1:
        base_model = torch.nn.parallel.DistributedDataParallel(
            base_model.to(device),
            device_ids=[local_rank],
            output_device=local_rank
        )
        print(f"#### [Rank {local_rank}] Model is wrapped with DDP")
    else:
        base_model.to(device)
    print("######################### Model is moved to device #########################")
    
    train_dataset, eval_dataset = create_dataset(args=script_args)
    print(f"#### train_dataset: {len(train_dataset)}")
    print(f"#### eval_dataset: {len(eval_dataset)}")
    original_columns = train_dataset.column_names
    print("########################## Successfully created datasets #########################")
    
    response_template_with_context = "<|assistant|>: "
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)
    print("##### response_template_ids", response_template_ids)
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
        return_tensors="pt"
    )
    print("######################### DataCollator is created #########################")
    
    training_args = SFTConfig(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        learning_rate=script_args.learning_rate,
        logging_steps=script_args.logging_steps,
        num_train_epochs=script_args.num_train_epochs,
        report_to=script_args.log_with,
        save_steps=script_args.save_steps,
        eval_steps=script_args.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=1,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_ratio=script_args.warmup_ratio,
        optim=script_args.optimizer_type,
        bf16=True if script_args.torch_dtype == "bfloat16" else False,
        remove_unused_columns=False,
        run_name="sft-c" if script_args.contextualize else "sft-nc",
        ddp_find_unused_parameters=False,
        dataset_text_field="text",
        max_seq_length=script_args.seq_length,
        dataloader_num_workers=script_args.num_workers,  # 데이터 로딩 워커 수 설정
        dataloader_pin_memory=True,  # 메모리 핀닝 활성화
    )
    print("######################### TrainingArguments is created #########################")
    
    callbacks = [ProgressCallback()]
    formatter = FormattingFunction(tokenizer.eos_token)
    
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # peft_config=peft_config,
        formatting_func=formatter.formatting_func_contextualize if script_args.contextualize \
            else formatter.formatting_func_plain,
        data_collator=data_collator,
        processing_class=tokenizer,
        args=training_args,
        callbacks=callbacks,
    )
    # convert to tensor
    trainer.train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    trainer.eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    print("######################### Trainer is set #########################")

    trainer.train()
    print("######################### Training is completed #########################")
    
    # to force the model to be saved only once in main process
    if not dist.is_initialized() or dist.get_rank() == 0:
        model_to_save = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        model_to_save.save_pretrained(script_args.output_dir)
        print(f"##### Model is saved to {script_args.output_dir}")
    
    if script_args.use_lora:
        output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
        model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
        # save the model: LoRA
        model_to_save.save_pretrained(output_dir)
        print(f"#### Model to save: {model_to_save}")

        del trainer
        # del model
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