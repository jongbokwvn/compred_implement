import os
import random
import torch
import torch.distributed as dist
from datetime import timedelta
from typing import Dict
from datasets import load_dataset, Dataset
from transformers import StoppingCriteria

# Class for Distributed Data Parallel (DDP)
class DDPManager:
    '''
    example:
        ddp_manager = DDPManager()
        if ddp_manager.is_main_process():
            print("Main process running")
        ddp_manager.cleanup()
    '''
    def __init__(self):
        self.rank, self.world_size, self.local_rank = self.setup_ddp()
        
    def setup_ddp(self):
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl", timeout=timedelta(seconds=7200))
            print(f"#### [Rank {rank}] Initialized DDP with world size {world_size}, using GPU {local_rank}")
            return rank, world_size, local_rank
        else:
            return 0, 1, 0
        
    def cleanup(self):
        if dist.is_initialized():
            dist.destroy_process_group()
        else:
            print("#### DDP is not initialized")
    
    def is_main_process(self):
        return self.rank == 0

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_key = key[len("module."):]
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

def rename_lora_keys(state_dict):
    new_state_dict = {}
    for old_key, value in state_dict.items():
        new_key = old_key
        if ".lora_A.weight" in old_key:
            new_key = old_key.replace(".lora_A.weight", ".lora_A.default.weight")
        if ".lora_B.weight" in old_key:
            new_key = old_key.replace(".lora_B.weight", ".lora_B.default.weight")
        if "lora_embedding_A" in old_key and ".default" not in old_key:
            new_key = old_key.replace("lora_embedding_A", "lora_embedding_A.default")
        if "lora_embedding_B" in old_key and ".default" not in old_key:
            new_key = old_key.replace("lora_embedding_B", "lora_embedding_B.default")
        if "gate_up_proj.lora_A" in old_key:
            new_key = old_key.replace("gate_up_proj.lora_A", "gate_up_proj.lora_A.default")
        if "gate_up_proj.lora_B" in old_key:
            new_key = old_key.replace("gate_up_proj.lora_B", "gate_up_proj.lora_B.default")
        if "down_proj.lora_A" in old_key:
            new_key = old_key.replace("down_proj.lora_A", "down_proj.lora_A.default")
        if "down_proj.lora_B" in old_key:
            new_key = old_key.replace("down_proj.lora_B", "down_proj.lora_B.default")
        if "qkv_proj.lora_A" in old_key:
            new_key = old_key.replace("qkv_proj.lora_A", "qkv_proj.lora_A.default")
        
        new_state_dict[new_key] = value
    return new_state_dict


def convert_state_dict_to_peft_format(state_dict, r=64):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith('.weight') and not any(x in key for x in ['.absmax', '.quant_map', '.quant_state']):
            # 기본 레이어 가중치
            base_key = key.replace('model.', 'base_model.model.model.')
            new_state_dict[base_key.replace('.weight', '.base_layer.weight')] = value
            
            # LORA A, B 가중치 생성
            if any(target in key for target in ['qkv_proj', 'o_proj', 'gate_up_proj', 'down_proj', 'embed_tokens']):
                shape = value.shape
                lora_A_key = base_key.replace('.weight', '.lora_A.default.weight')
                lora_B_key = base_key.replace('.weight', '.lora_B.default.weight')
                
                # LORA 가중치 초기화
                new_state_dict[lora_A_key] = torch.randn((r, shape[1])) * 0.02
                new_state_dict[lora_B_key] = torch.randn((shape[0], r)) * 0.02

    return new_state_dict

# formatting function
class FormattingFunction:
    def __init__(self, eos_token):
        self.eos_token = eos_token
    def formatting_func_plain(self, example):
        '''
        <|domain|>: {domain}
        <|user|>
            {title}
            {history}
        <|assistant|>: {human_ref_A}
        '''
        instruction_response = f"<|domain|> Reddit\n<|user|>\n{example['title']}\n{example['history']}\n<|assistant|>: {example['human_ref_A'] if example['labels'] == 1 else example['human_ref_B']}{self.eos_token}"
        return instruction_response

    def formatting_func_contextualize(self, example):
        '''
        <|domain|>: Reddit
        <|user|>
            {title}
            {history}
        <|assistant|>: {human_ref_A}
        '''
        instruction_response = f"<|domain|> {example['domain']}\n<|user|>\n{example['title']}\n{example['history']}\n<|assistant|>: {example['human_ref_A'] if example['labels'] == 1 else example['human_ref_B']}{self.eos_token}"
        return instruction_response
    
    def formatting_func_aot_plain(self, example):
        return {
            'instruction': f"<|domain|>: Reddit\n<|user|>\n{example['title']}\n{example['history']}",
            'response': f"{example['human_ref_A'] if example['labels'] == 1 else example['human_ref_B']}"
        }
    
    def formatting_func_aot_contextualize(self, example):
        return {
            'instruction': f"<|domain|>: {example['domain']}\n<|user|>\n{example['title']}\n{example['history']}",
            'response': f"{example['human_ref_A'] if example['labels'] == 1 else example['human_ref_B']}"
        }
            
def create_dataset(args):
    data_path = args.dataset_name
    subset = args.subset
    subreddit_name = args.subreddit_name
    
    # filtering function
    def load_and_process_split(split):
        dataset = load_dataset(data_path, subset, split=split, streaming=args.streaming)
        # filter with domain and score_ratio        
        if subreddit_name == "none":
            filtered_dataset = dataset.filter(
                lambda x: x['score_ratio'] >= args.score_ratio_threshold, 
                num_proc=args.num_workers
            )
        else:
            filtered_dataset = dataset.filter(
                lambda x: x['domain'] == subreddit_name and x['score_ratio'] >= args.score_ratio_threshold, 
                num_proc=args.num_workers
            )
        if args.num_examples_per_post:
            df = filtered_dataset.to_pandas()
            df = df.groupby("post_id").apply(
                lambda x: x.sample(n=min(args.num_examples_per_post, len(x)))
            )
            df = df.sample(n=len(df))
            filtered_dataset = Dataset.from_pandas(df)
        return filtered_dataset
    
    train_dataset = load_and_process_split('train_pref')
    valid_dataset = load_and_process_split('validation_pref')
    
    if len(train_dataset) == 0 or len(valid_dataset) == 0:
        raise ValueError(f"#### Filtered dataset is empty. Check subreddit_name={subreddit_name} and score_ratio_threshold={args.score_ratio_threshold}")
    
    print(f"#### Train dataset size: {len(train_dataset)}")
    print(f"#### Validation dataset size: {len(valid_dataset)}")
    
    return train_dataset, valid_dataset

class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        # stop_id_sequences should be a list, not a string
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences
        self.all_done = None    # track the stopping of generated samples

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # sequences_should_be_stopped = []
        if self.all_done is None:
            self.all_done = [False for _ in range(input_ids.shape[0])]
        for i in range(input_ids.shape[0]):
            if not self.all_done[i]:
                for stop_sequence in self.stop_sequences:
                    #print(stop_sequence, input_ids[i][-len(stop_sequence):].tolist())
                    # if the last token sequence of the generated sample matches stop_sequence, set self.all_done[i] = True to stop the generation of the sample
                    if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                        self.all_done[i]
                        break
            # sequences_should_be_stopped.append(sequence_should_be_stopped)
            # print(sequence_should_be_stopped)
        return all(self.all_done)
    
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def subsample(dataset, ratio_thresh, examples_per_post):
    df = dataset.to_pandas()
    df = df[df["score_ratio"] >=  ratio_thresh]
    df = df.groupby("post_id").apply(
        lambda x: x.sample(n=min(examples_per_post, len(x)))
    )
    df = df.sample(n=len(df))
    return Dataset.from_pandas(df)

def get_paired_dataset(
    script_args,
    data_path: str,
    split: str = "train_pref",
    subset: str = "history",
    sanity_check: bool = False,
    num_proc=24,
    tokenizer=None,
) -> Dataset:

    tmp_dataset = load_dataset(data_path, subset, split=split)
    if script_args.subreddit_name != "none":
        dataset = tmp_dataset.filter(lambda x: x['domain'] == script_args.subreddit_name)
    else:
        dataset = tmp_dataset
    print(f"####raw dataset\n{dataset}")
   
    cols_to_remove = dataset.column_names
    cols_to_remove.remove("domain")
    cols_to_remove.remove("history")
    cols_to_remove.remove("human_ref_A")
    cols_to_remove.remove("human_ref_B")
    cols_to_remove.remove("labels")
    cols_to_remove.remove("score_ratio")

    dataset.remove_columns(cols_to_remove)

    print(f"#### Original {split} data size: {len(dataset)}")
    dataset = subsample(dataset, script_args.score_ratio_threshold, script_args.num_examples_per_post)
    print(f"#### Filtered {split} data with >{script_args.score_ratio_threshold} score ratio and {script_args.num_examples_per_post} comment pairs per post: {len(dataset)}")

    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return_object = {"prompt": [], "chosen": [], "rejected": []}
        
        for subreddit, title, question, response_j, response_k, label in zip(samples['domain'], samples["title"], samples["history"], samples["human_ref_A"], samples["human_ref_B"], samples['labels']):
            subreddit = subreddit#.split("_")[0]
            
            response_j += tokenizer.eos_token
            response_k += tokenizer.eos_token

            instruction = "<|domain|>\n{domain}\n<|user|>{title}\n{post}\n<|assistant|>\n"
            if script_args.contextualize:
                domain = "r/"+subreddit
            else:
                domain = "Reddit"
                
            if label == 0:
                response_j, response_k = response_k, response_j
            
            # prompt = instruction + question + " \n\n COMMENT: "
            prompt = instruction.format(domain=domain, title=title, post=question)
            return_object['prompt'].append(prompt)
            return_object['chosen'].append(response_j)
            return_object['rejected'].append(response_k)

        return return_object
    
    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

def compute_grad_norm(model):
    """
    model.parameters()에 누적되어 있는 p.grad로부터
    Grad norm을 직접 계산하는 코드.
    """
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm_sq += param_norm.item() ** 2
    return total_norm_sq ** 0.5