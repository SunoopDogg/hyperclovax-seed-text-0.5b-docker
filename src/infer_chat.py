import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

CWD = os.getcwd()

DEFAULTS = {
    "model": f"{CWD}/HyperCLOVAX-SEED-Text-Instruct-0.5B",
    "max_new_tokens": 128,
    "repetition_penalty": 1.2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    "do_sample": True,
    "top_p": 0.9,
    "temperature": 0.7,
}


def load_settings():
    """환경변수 INFER_CONFIG로 지정된 경로나 ./infer_config.json이 있으면 읽고, 없으면 DEFAULTS 사용."""
    cfg_path = os.environ.get("INFER_CONFIG", "infer_config.json")
    config = DEFAULTS.copy()
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            # DEFAULTS 키만 반영
            for k in DEFAULTS.keys():
                if k in user_cfg and user_cfg[k] is not None:
                    config[k] = user_cfg[k]
        except Exception as e:
            print(f"[WARN] 설정 파일 로드 실패: {e}. 기본값으로 진행합니다.")
    return config


def run_inference(settings):
    model_id = settings["model"]

    prompt = input("프롬프트를 입력하세요: ")

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else None),
    )

    chat = [
        {"role": "tool_list", "content": ""},
        {"role": "system", "content": "- AI 언어모델의 이름은 \"CLOVA X\" 이며 네이버에서 만들었다."},
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    gen_kwargs = dict(
        max_new_tokens=int(settings["max_new_tokens"]),
        repetition_penalty=float(settings["repetition_penalty"]),
        do_sample=bool(settings["do_sample"]),
        top_p=float(settings["top_p"]),
        temperature=float(settings["temperature"]),
        eos_token_id=tokenizer.eos_token_id,
        tokenizer=tokenizer,
    )

    output_ids = model.generate(**inputs, **gen_kwargs)
    print(tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=False))


def main():
    settings = load_settings()
    run_inference(settings)


if __name__ == "__main__":
    main()
