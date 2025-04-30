from tqdm import tqdm
import os
import json
from arguments import StanceClassifierArguments
from transformers import HfArgumentParser, pipeline
from utils import set_seed
import torch

# Zero-shot stance classification
def zero_shot_stance(classifier, response):
    result = classifier(response, candidate_labels=["conservative", "askaliberal"])
    conservative_score = result["scores"][result["labels"].index("conservative")]
    liberal_score = result["scores"][result["labels"].index("askaliberal")]

    if conservative_score > liberal_score:
        return [{"label": "CONSERVATIVE", "score": conservative_score}]
    else:
        return [{"label": "LIBERAL", "score": liberal_score}]

def load_json_in_8line_chunks(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()  # 파일 전체를 줄 단위로 읽어옵니다.

    objects = []
    # range(0, len(lines), 8)을 사용하면 8개 단위로 끊어서 순회할 수 있습니다.
    for i in range(0, len(lines), 8):
        chunk = lines[i:i+8]

        # 남은 줄이 8줄보다 적으면(파일 끝), break 또는 원하는 로직 처리
        if len(chunk) < 8:
            break

        # 8줄을 합쳐 하나의 문자열로 만든 뒤, JSON으로 파싱
        chunk_str = ''.join(chunk).strip()
        obj = json.loads(chunk_str)
        objects.append(obj)

    return objects

def main():
    parser = HfArgumentParser(StanceClassifierArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    model = script_args.model_name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_file = script_args.input_dir
    output_path_name = script_args.output_dir
    os.makedirs(output_path_name, exist_ok=True)
    input_file_name = script_args.input_file_name
    output_file_name = os.path.join(output_path_name, f"{input_file_name}.jsonl")

    # Load stance classifier
    classifier = pipeline("zero-shot-classification", model=model, device=device)
    print("########################## Classifier is Loaded ##########################")
    
    data = load_json_in_8line_chunks(input_file)
    print(f"#### len: {len(data)}")
    print(f"########################## {len(data)} data is loaded ##########################")

    with open(output_file_name, "w", encoding="utf-8") as f:
        for idx, item in tqdm(enumerate(data), total=len(data)):
            domain = item["domain"]
            title = item["title"]
            post = item["post"]
            response_dict = item["response"]

            # 포스트 텍스트 구성
            post_text = f"<|domain|>{domain}<|user|>\n{title}\n{post}"

            for response_id, response_text in response_dict.items():
                # 응답을 포함한 전체 텍스트 구성
                full_text = post_text + "\n<|assistant|>\n" + response_text

                # 모델 실행
                result = zero_shot_stance(classifier, full_text)

                if result[0]["label"] == "CONSERVATIVE":
                    conservative = result[0]["score"]
                    liberal = 1 - result[0]["score"]
                else:
                    conservative = 1 - result[0]["score"]
                    liberal = result[0]["score"]

                # 결과 저장
                f.write(f"{idx}-{response_id} conservative: {conservative:.4f} liberal: {liberal:.4f}\n")
    print(f"########################## {output_file_name} is saved ##########################")

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    set_seed(42)
    main()