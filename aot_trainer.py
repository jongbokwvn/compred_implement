'''
build based on sft_trainer.py
'''

import os
import gc
import json
import wandb
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as CosineAnnealingLR
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import logging
import warnings
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    get_cosine_schedule_with_warmup
)
from transformers.trainer_callback import ProgressCallback
from tqdm import tqdm
from bitsandbytes.optim import AdamW32bit as PageAdamW32bit
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from datasets import load_dataset, Dataset
from safetensors.torch import load_file
from arguments import AOTTrainerArguments
from utils import (
    DDPManager, create_dataset, FormattingFunction,
    subsample, remove_module_prefix, rename_lora_keys,
    set_seed, compute_grad_norm
)
from constants import HF_TOKEN
from huggingface_hub.commands.user import login; login(token=HF_TOKEN)

### ABOUT Loss function: AOT (Alignment via Optimal Transport)
def fsd_loss(logits, target_distribution):
    """
    FSD (First Order Stochastic Dominance) 손실:
    - CDF-based hinge loss를 사용한다는 점에서 log-ratio를 활용하는 방법이 아님
    - logits: [B, T, vocab_size]
    - target_distribution: [B, T, vocab_size] 또는 [vocab_size] (배치/시간축에 브로드캐스트 가능)
    
    generated 확률 분포와 target 확률 분포의 누적분포함수를 각각 계산한 후,
    generated의 CDF가 target의 CDF를 초과하는(즉, hinge violation) 경우에 대해 패널티를 부여.
    
    여기서는 간단히 각 배치 및 시퀀스별 평균 위반 정도를 계산합니다.
    """
    B, T, V = logits.size()
    # generated 확률 분포 계산
    prob_generated = F.softmax(logits, dim=-1)  # [B, T, V]
    # generated 누적분포함수(CDF)
    cdf_generated = torch.cumsum(prob_generated, dim=-1)
    
    # target_distribution이 [B, T, V]가 아니면, 필요에 따라 확률 분포로 변환 후 브로드캐스트
    if target_distribution.dim() == 1:
        # [vocab_size] -> [1, 1, V] -> [B, T, V]
        target_distribution = target_distribution.view(1, 1, V).expand(B, T, V)
    elif target_distribution.dim() == 2:
        # [B, V] -> [B, 1, V] -> [B, T, V]
        target_distribution = target_distribution.unsqueeze(1).expand(B, T, V)
    else:
        # 이미 [B, T, V]인 경우
        pass
    # target의 누적분포함수(CDF)
    cdf_target = torch.cumsum(target_distribution, dim=-1)
    
    # FSD 위반: generated CDF가 target CDF를 초과하는 부분에 대해 패널티 적용
    # (예시에서는 단순히 양의 차이를 평균내는 hinge loss 형태로 구현)
    violation = F.relu(cdf_generated - cdf_target)
    loss = violation.mean()
    return loss

class AlignmentLossFSD(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, logits, target_distribution):
        B, T, V = logits.size()
        prob_generated = F.softmax(logits, dim=-1)  # [B, T, V]
        cdf_generated = torch.cumsum(prob_generated, dim=-1)
        
        # vocab_size -> broadcast
        target_distribution = target_distribution.view(1, 1, V).expand(B, T, V)
        cdf_target = torch.cumsum(target_distribution, dim=-1)
        violation = F.relu(cdf_generated - cdf_target)
        return violation.mean()
 
def main():
    # parsing arguments
    parser = HfArgumentParser(AOTTrainerArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    # Setting NCCL timeout
    os.environ["NCCL_TIMEOUT"] = "3600"  # 1시간으로 설정
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"  # 새로운 환경변수 사용
    os.environ["NCCL_BLOCKING_WAIT"] = "1"  # 블로킹 대기 모드 활성화
    os.environ["NCCL_P2P_DISABLE"] = "1"  # P2P 비활성화
    os.environ["NCCL_IB_DISABLE"] = "1"  # IB 비활성화
    
    # initialize for DDP
    ddp_manager = DDPManager()
    rank, world_size, local_rank = ddp_manager.rank, ddp_manager.world_size, ddp_manager.local_rank
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    print("########################## Device is set #########################")
    
    # setting output directory
    path_name = os.path.join(
        script_args.output_dir, "aot-c", script_args.subset, script_args.subreddit_name, script_args.model_name.replace("/", "-")
    ) if script_args.contextualize \
        else os.path.join(
            script_args.output_dir, "aot-nc", script_args.subset, script_args.subreddit_name, script_args.model_name.replace("/", "-")
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
                device_map="auto",
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
    
    target_dist_path = script_args.target_dist_path
    try:
        with open(target_dist_path, 'rb') as f:
            target_distribution = pickle.load(f)
    except FileNotFoundError:
        ddp_manager.cleanup()
        raise FileNotFoundError(f"Target distribution file not found at {target_dist_path}")
    except Exception as e:
        print(f"Error loading target distribution: {str(e)}")
        ddp_manager.cleanup()
        return
    
    target_distribution = torch.tensor(target_distribution, dtype=script_args.torch_dtype)
    print("########################## Target Distribution is loaded #########################")

    train_dataset, eval_dataset = create_dataset(script_args)
    original_columns = train_dataset.column_names
    formatter = FormattingFunction(tokenizer.eos_token)
    train_dataset = train_dataset.map(
        formatter.formatting_func_aot_contextualize if script_args.contextualize \
            else formatter.formatting_func_aot_plain,
        num_proc=script_args.num_workers,
        remove_columns=original_columns
    )
    
    print(f"#### train_dataset: {len(train_dataset)}")
    print(f"#### eval_dataset: {len(eval_dataset)}")
    print("########################## Dataset is loaded #########################")

    # initialize wandb to report
    if ddp_manager.is_main_process():
        wandb.init(
            project="AOT_modified",
            config={
                "model": "microsoft/phi-3.5-mini-instruct",
                "subreddit_name": script_args.subreddit_name,
                "num_epochs": script_args.num_train_epochs,
                "batch_size": script_args.per_device_train_batch_size,
                "learning_rate": script_args.learning_rate,
                "weight_decay": script_args.weight_decay,
                "warmup_ratio": script_args.warmup_ratio,
                "gradient_accumulation_steps": script_args.gradient_accumulation_steps,
                "lambda_align": script_args.lambda_align,
                "optimizer_type": script_args.optimizer_type,
                "lr_scheduler_type": script_args.lr_scheduler_type
            }
        )
    else:
        wandb.init(mode="disabled")
        
    alignment_loss_fn = AlignmentLossFSD()
    
    if script_args.gradient_checkpointing:
        base_model.module.gradient_checkpointing_enable()
        
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(
        train_dataset,
        batch_size=script_args.per_device_train_batch_size,
        sampler=train_sampler,
        num_workers=script_args.num_workers,
        pin_memory=True
    )
    print("######################### DataLoader is set #########################")
    
    if script_args.optimizer_type == "paged_adamw_32bit":
        optimizer = PageAdamW32bit(
            base_model.parameters(),
            lr=script_args.learning_rate,
            weight_decay=script_args.weight_decay
        )
    else:
        raise ValueError("#### Please check your optimizer_type")
    print("######################### Optimizer is set #########################")
    
    num_update_steps_per_epoch = len(dataloader) // script_args.gradient_accumulation_steps
    num_training_steps = num_update_steps_per_epoch * script_args.num_train_epochs
    print(f"#### num_training_steps: {num_training_steps}")
    num_warmup_steps = int(num_training_steps * script_args.warmup_ratio)
    print(f"#### num_warmup_steps: {num_warmup_steps}")
    
    if script_args.lr_scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    else:
        raise ValueError("#### Please check your lr_scheduler_type")
    print("######################### Scheduler is set #########################")
    
    global_step = 0
    for epoch in range(script_args.num_train_epochs):
        train_sampler.set_epoch(epoch)
        
        base_model.train()
        epoch_loss = 0.0
        step_count = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{script_args.gradient_accumulation_steps * script_args.num_train_epochs}", disable=(rank != 0))
        
        for batch_idx, batch in enumerate(pbar):
            merged_texts = [
                p + "\n" + a
                for p, a in zip(batch['instruction'], batch['response'])
            ]
            inputs = tokenizer(
                merged_texts,
                return_tensors='pt',
                padding=True,
                padding_side='left',
                truncation=True,
                max_length=script_args.seq_length
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            # prompt 길이 계산
            prompt_encoded = tokenizer(
                batch['instruction'],
                return_tensors='pt',
                padding=True,
                padding_side='left',
                truncation=True,
                max_length=script_args.max_prompt_length
            )
            prompt_lengths = prompt_encoded['input_ids'].size(1)
            
            labels = input_ids.clone()
            
            # prompt 영역 마스킹
            labels[:, :prompt_lengths] = -100
            
            outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # CE Loss
            ce_loss = outputs.loss
            
            # AOT Loss
            all_logits = outputs.logits
            answer_logits = all_logits[:, prompt_lengths:, :]
            alignment_loss = alignment_loss_fn(answer_logits, target_distribution.to(rank))            
            
            # total loss
            total_loss_value = ce_loss + script_args.lambda_align * alignment_loss
            epoch_loss += total_loss_value.item()
            
            parameters = [p for p in base_model.parameters() if p.requires_grad]
            
            with torch.no_grad():
                ce_grads = torch.autograd.grad(ce_loss, parameters, retain_graph=True)
                ce_grad_norm = torch.sqrt(sum(grad.norm(2)**2 for grad in ce_grads))
                
                align_grads = torch.autograd.grad(alignment_loss, parameters, retain_graph=True)
                align_grad_norm = torch.sqrt(sum(grad.norm(2)**2 for grad in align_grads))
            
            scaled_loss = total_loss_value / script_args.gradient_accumulation_steps
            scaled_loss.backward()

            # 실제 optimizer step 시점
            if (batch_idx + 1) % script_args.gradient_accumulation_steps == 0:
                # backward 완료된 gradient_norm 측정
                total_grad_norm = torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)               
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                step_count += 1
                
                if (global_step % script_args.logging_steps == 0) and (rank == 0):
                    wandb.log({
                        'global_step': global_step,
                        'ce_loss': ce_loss.item(),
                        'alignment_loss': alignment_loss.item(),
                        'train_loss': total_loss_value,
                        'ce_grad_norm': ce_grad_norm,
                        'align_grad_norm': align_grad_norm,
                        'total_grad_norm': float(total_grad_norm),
                        'learning_rate': optimizer.param_groups[0]['lr']
                    })
                
                pbar.set_postfix({
                    'global_step': global_step,
                    'ce_loss': f"{ce_loss.item():.4f}",
                    'alignment_loss': f"{alignment_loss.item():.4f}",
                    'train_loss': f"{total_loss_value:.4f}",
                    'ce_grad_norm': f"{ce_grad_norm:.4f}",
                    'align_grad_norm': f"{align_grad_norm:.4f}",
                    'total_grad_norm': f"{float(total_grad_norm):.4f}",
                    'learning_rate': f"{optimizer.param_groups[0]['lr']: .4f}"
                })
                
        if rank == 0:
            print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}")
            wandb.log({
                "epoch_loss": epoch_loss,
                "ce_loss": ce_loss.item(),
                "alignment_loss": alignment_loss.item(),
                "ce_grad_norm": ce_grad_norm,
                "align_grad_norm": align_grad_norm,
                "total_grad_norm": float(total_grad_norm),
                "learning_rate": optimizer.param_groups[0]['lr']
            })
    print("######################### Training is finished #########################")
    
    # HF 스타일로 최종 체크포인트 저장
    # 모든 epoch 종료 후 마지막 저장 (rank=0만)
    if rank == 0:
        if isinstance(base_model, torch.nn.parallel.DistributedDataParallel):
            model_to_save = base_model.module
        else:
            model_to_save = base_model
            
        if script_args.use_lora:
            lora_target_modules = ['qkv_proj', 'o_proj', 'gate_up_proj', 'down_proj', 'embed_tokens']
            lora_config = LoraConfig(
                r=script_args.lora_r,
                lora_alpha=script_args.lora_alpha,
                lora_dropout=script_args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=lora_target_modules
            )
            peft_model_to_save = PeftModel(model_to_save, lora_config)
            peft_model_to_save.save_pretrained(
                os.path.join(script_args.output_dir, 'lora_checkpoint')
            )
            tokenizer.save_pretrained(
                os.path.join(script_args.output_dir, 'lora_checkpoint')
            )
        
        final_hf_dir = os.path.join(script_args.output_dir, 'final_merged_checkpoint')
        os.makedirs(final_hf_dir, exist_ok=True)
        
        model_to_save.save_pretrained(final_hf_dir)
        tokenizer.save_pretrained(final_hf_dir)
        
        torch.save(optimizer.state_dict(), os.path.join(final_hf_dir, 'optimizer.pt'))
        torch.save(scheduler.state_dict(), os.path.join(final_hf_dir, 'scheduler.pt'))
        with open(os.path.join(final_hf_dir, 'training_args.bin'), 'wb') as f:
            pickle.dump(script_args, f)
        
        print(f"[Rank 0] Final model saved to {final_hf_dir}")
            
        # final_ckpt = os.path.join(script_args.output_dir, f"aot_cons_dist_{script_args.subreddit_name}.pt")
        # torch.save(base_model.module.state_dict(), final_ckpt)
        # print(f"[Rank 0] Final model saved to {final_ckpt}")

    wandb.finish()
    
    if world_size > 1:
        dist.barrier()
        ddp_manager.cleanup()
    gc.collect()
    torch.cuda.empty_cache()
    
if __name__ == "__main__":
    # GPU Settings
    # os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'
    set_seed(42)
    main()