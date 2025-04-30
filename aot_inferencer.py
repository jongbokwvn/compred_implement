import os
import gc
import json
import time
import torch
import torch.distributed as dist
import logging
import warnings
import random
import pysbd
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional
from datasets import load_dataset
from dataclasses import dataclass, field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    StoppingCriteria
)
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM, PeftModel
from constants import HF_TOKEN
from huggingface_hub.commands.user import login; login(token=HF_TOKEN)
from safetensors.torch import load_file
import wandb

from arguments import AOTInferencerArguments
from utils import (
    DDPManager,
    KeyWordsCriteria,
    set_seed,
    remove_module_prefix,
    rename_lora_keys,
    convert_state_dict_to_peft_format
)

def main():
    # parsing arguments
    parser = HfArgumentParser(AOTInferencerArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    # initialize DDP
    ddp_manager = DDPManager()
    rank, world_size, local_rank = ddp_manager.rank, ddp_manager.world_size, ddp_manager.local_rank
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    print(f"#### rank: {rank}, world_size: {world_size}, local_rank: {local_rank}")
    print("######################### Device is set #########################")
    
    # setting output directory
    output_path_name = os.path.join(
        script_args.output_dir, "aot-c", script_args.subset, script_args.subreddit_name, script_args.model_name.replace("/", "-") if script_args.contextualize \
            else script_args.output_dir, "aot-nc", script_args.subset, script_args.subreddit_name, script_args.model_name.replace("/", "-")
    )
    os.makedirs(output_path_name, exist_ok=True)
    output_file_name = os.path.join(output_path_name, f"aot_conservative_{local_rank}.jsonl")
    print("######################### Output directory is set #########################")
    
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
    
    stop_id_sequences = []
    for stopstring in ["\n<|", " <|", "<|"]:
        stop_id_sequences.append(tokenizer.encode(stopstring, add_special_tokens=False)[2:])
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
    
    # LORA 체크포인트 로딩
    if script_args.model_state_path:
        checkpoint_path = script_args.model_state_path
        lora_target_modules = ['qkv_proj', 'o_proj', 'gate_up_proj', 'down_proj', 'embed_tokens']
        
        # final_merged_checkpoint인 경우
        if os.path.basename(checkpoint_path) == "final_merged_checkpoint":
            print(f"#### Loading merged checkpoint from {checkpoint_path}")
            base_model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                trust_remote_code=True,
                torch_dtype=script_args.torch_dtype,
                device_map="auto"
            )
        # LORA checkpoint인 경우
        else:
            print(f"#### Loading LORA checkpoint from {checkpoint_path}")
            if os.path.exists(os.path.join(checkpoint_path, "adapter_model.safetensors")):
                # PEFT/LORA 설정 로드
                with open(os.path.join(checkpoint_path, "adapter_config.json"), 'r') as f:
                    adapter_config = json.load(f)
                
                lora_config = LoraConfig(
                    r=adapter_config.get('r', script_args.lora_r),
                    lora_alpha=adapter_config.get('lora_alpha', script_args.lora_alpha),
                    lora_dropout=adapter_config.get('lora_dropout', script_args.lora_dropout),
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=lora_target_modules
                )
                
                # PEFT 모델 초기화
                base_model = get_peft_model(base_model, lora_config)
                
                # safetensors에서 가중치 로드
                state_dict = load_file(os.path.join(checkpoint_path, "adapter_model.safetensors"))
                state_dict = remove_module_prefix(state_dict)
                state_dict = rename_lora_keys(state_dict)
                
                print(f"#### Loaded state_dict from {checkpoint_path} with {len(state_dict.keys())} keys.")
                some_keys = list(state_dict.keys())[:10]  # 앞 몇 개만 출력
                print(f"#### Some of the keys: {some_keys} ...")

                missing_keys, unexpected_keys = base_model.load_state_dict(state_dict, strict=False)
                # print(f"#### Missing keys: {missing_keys}")
                print(f"#### Unexpected keys: {unexpected_keys}")
                
            else:
                raise ValueError(f"Cannot find adapter_model.safetensors in {checkpoint_path}")
    print("######################### Model is loaded #########################")
    
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
    
    # preprocessing for remove duplicates in test_prompts
    tmp_ds = load_dataset(script_args.dataset_name, script_args.subset+"_test_prompts", split="test_prompts")
    if script_args.subreddit_name != "none":
        ds = tmp_ds.filter(lambda x: x['domain'] == script_args.subreddit_name, num_proc=script_args.num_workers)
    else:
        ds = tmp_ds
    
    print(f"#### number of test examples before removing duplicates: {len(ds)}")
    cols_to_remove = ds.column_names
    cols_to_remove.remove('title')
    cols_to_remove.remove('history')
    cols_to_remove.remove('domain')
    cols_to_remove.remove('post_id')
    ds = ds.remove_columns(cols_to_remove)
    df = ds.to_pandas()
    
    df['unique_key'] = df['title'] + "_" + df['post_id'] + "_" + df['domain'] + "_" + df['history']
    df = df.drop_duplicates(subset=['unique_key']).reset_index(drop=True)
    # df = df[:600]
    print(f"#### number of test examples after removing duplicates: {len(df)}")
    
    # allocate data to each GPU
    splits = np.array_split(df, world_size)
    df = pd.DataFrame(splits[local_rank]).reset_index(drop=True)
    print("######################### Dataset is loaded and preprocessed #########################")

    logging.info(f"#### Done loading data with {ds.shape[0]} entries!")
    
    batch_size = script_args.batch_size
    x, y = 0, 0
    st = time.time()
    emptylines, text_batch, posttext_batch, title_batch, domains = [], [], [], [], []
    all_domains = set()
    
    for i, row in df.iterrows():
        domain = row['domain']
        all_domains.add(domain)
    all_domains = list(all_domains)
    
    print("######################### Inference is started #########################")
    fout = open(output_file_name, "w")
    
    segmenter = pysbd.Segmenter(language="en", clean=False)
    with torch.no_grad():
        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            if (i == 0 or len(text_batch) < batch_size):
                posttext_batch.append(row["history"])
                title_batch.append(row["title"])
                subreddit = row['domain']
                if script_args.randomize_context:
                    new_subreddit = random.choice(all_domains)
                    dcount = 0
                    while new_subreddit == subreddit and dcount < 20:
                        new_subreddit = random.choice(all_domains)
                        dcount += 1
                    subreddit = new_subreddit
                domains.append(subreddit)
                instruction = "<|domain|>: {domain}\n<|user|>\n{title_and_post}\n<|assistant|>: "
                post=row["history"]
                title=row["title"]
                title_and_post=f"{title}\n{post}"
                
                sentences = []
                slack = 1024    # split by max_length
                for s in segmenter.segment(title_and_post):
                    l = len(tokenizer(s).input_ids)
                    slack -= l
                    
                    if slack > 0:
                        sentences.append(s)
                title_and_post = "".join(sentences)
            
                if script_args.contextualize:
                    domain = subreddit
                else:
                    domain = "Reddit"
                text_batch.append(instruction.format(domain=domain, title_and_post=title_and_post))

                if (len(text_batch) < batch_size and i < df.shape[0]-1):
                    continue
            
            # to measure time per sentence
            st = time.time()
            
            tokens = tokenizer(
                text_batch,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                truncation=True,
                max_length=script_args.seq_length
            ).to(device)
            
            st = time.time()
            
            text_outputs = [{} for _ in range(len(text_batch))]
            for beam_size in [1]:
                generation_kwargs = {"num_beams": beam_size, "do_sample": False, "repetition_penalty": 1.1}
                outputs = base_model.module.generate(**tokens, max_new_tokens=script_args.seq_length,
                                         stopping_criteria=[KeyWordsCriteria(stop_id_sequences)],
                                         **generation_kwargs, use_cache=False) if hasattr(base_model, 'module') else base_model.generate(**tokens,
                                                                                                              max_new_tokens=script_args.seq_length,
                                                                                                              stopping_criteria=[KeyWordsCriteria(stop_id_sequences)],
                                                                                                              **generation_kwargs, use_cache=False)
                for k, output in enumerate(outputs):
                    prompt_tokens = tokenizer(text_batch[k], return_tensors='pt')['input_ids'][0]
                    prompt_length = prompt_tokens.size(0)
                    generated_tokens = output[prompt_length:]
                    full_text_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    text_outputs[k][f'{beam_size}'] = full_text_output
                    
            file_write = [json.dumps({'title': title, 'post': post, 'domain': domain, 'response': response}, indent=4, ensure_ascii=False)
                        for title, post, domain, response in zip(title_batch, posttext_batch, domains, text_outputs)]
            fout.write("\n".join(file_write) + "\n")
            fout.flush()
            
            x += len(text_outputs)
            y += len(file_write)
            text_batch, posttext_batch, title_batch, domains = [], [], [], []
            
            if i % 50 == 0:
                print(f"done total {y} inputs, took {(time.time()-st)/(x)} seconds per sentence since last log", flush=True)
                x = 0
    
    print(f"#### wrote {y} lines")
    
    fout.close()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    ddp_manager.cleanup()
    
if __name__ == "__main__":
    # GPU Setting
    set_seed(42)
    main()