import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LLM_PATH = "checkpoints/rl_grpo/grpo_8_2_iter1_np/checkpoint-480"
PROMPT = (
"""You are an expert in traffic management.

A traffic light regulates a four-way intersection with northern, southern, eastern, and western approaches, each containing two lanes: one for through traffic and one for left-turns. Each lane is further divided into three segments. Segment 1 is the closest to the intersection. Segment 2 is in the middle. Segment 3 is the farthest. In a lane, there may be early queued vehicles and approaching vehicles traveling in different segments. Early queued vehicles have already arrived at the intersection and await passage permission. Approaching vehicles will arrive at the intersection in the future.

The traffic light has 4 signal phases. Each signal relieves vehicles' flow in a group of two specific lanes.

Available signal phases:
- ETWT: Eastern and western through lanes.
- NTST: Northern and southern through lanes.
- ELWL: Eastern and western left-turn lanes.
- NLSL: Northern and southern left-turn lanes.

Current intersection state:
Signal: ETWT
Relieves: Eastern and western through lanes.
- Early queued: 0 (East), 1 (West), 1 (Total)
- Segment 1: 0 (East), 0 (West), 0 (Total)
- Segment 2: 1 (East), 0 (West), 1 (Total)
- Segment 3: 0 (East), 2 (West), 2 (Total)
- Neighbor incoming totals: 2 (East), 1 (West), 3 (Known total), 2/2 available

Signal: NTST
Relieves: Northern and southern through lanes.
- Early queued: 0 (North), 0 (South), 0 (Total)
- Segment 1: 0 (North), 0 (South), 0 (Total)
- Segment 2: 0 (North), 0 (South), 0 (Total)
- Segment 3: 1 (North), 0 (South), 1 (Total)
- Neighbor incoming totals: NA (North), 3 (South), 3 (Known total), 1/2 available

Signal: ELWL
Relieves: Eastern and western left-turn lanes.
- Early queued: 0 (East), 0 (West), 0 (Total)
- Segment 1: 0 (East), 0 (West), 0 (Total)
- Segment 2: 0 (East), 0 (West), 0 (Total)
- Segment 3: 0 (East), 1 (West), 1 (Total)
- Neighbor incoming totals: 2 (East), 1 (West), 3 (Known total), 2/2 available

Signal: NLSL
Relieves: Northern and southern left-turn lanes.
- Early queued: 0 (North), 0 (South), 0 (Total)
- Segment 1: 0 (North), 0 (South), 0 (Total)
- Segment 2: 0 (North), 0 (South), 0 (Total)
- Segment 3: 0 (North), 0 (South), 0 (Total)
- Neighbor incoming totals: NA (North), 3 (South), 3 (Known total), 1/2 available

The state description above lists:
- The group of lanes relieved under each traffic light phase.
- The number of early queued vehicles in the allowed lanes of each signal.
- The number of approaching vehicles in different segments of the allowed lanes of each signal.
- Neighbor incoming totals from adjacent intersections for each phase.
- `NA` means that adjacent side has a virtual/missing neighbor and is excluded from `Known total`.

Question:
Which is the most effective traffic signal that will most significantly improve the traffic condition during the next phase?

Note:
- Traffic congestion is primarily dictated by early queued vehicles, with the most significant impact.
- You must pay the most attention to lanes with long queue lengths.
- It is not urgent to consider vehicles in distant segments, since they are unlikely to reach the intersection soon.

Requirements:
- Think step by step.
- You can only choose one of the signals listed above.
- Step 1: Provide a brief analysis identifying the optimal traffic signal.
- Step 2: After finishing the analysis, answer with your chosen signal.
- Include exactly one final decision tag in this format: <signal>PHASE</signal>, where PHASE is one of: ETWT, NTST, ELWL, NLSL.
"""
)

MAX_NEW_TOKENS = 1024


def _is_lora_adapter_path(model_path: str) -> bool:
    return os.path.isdir(model_path) and os.path.isfile(
        os.path.join(model_path, "adapter_config.json")
    )


def _load_model_and_tokenizer(llm_path: str):
    model_path = os.path.normpath(os.path.expanduser(llm_path))
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LLM_PATH does not exist: {llm_path}")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if _is_lora_adapter_path(model_path):
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError(
                "LLM_PATH points to a LoRA adapter, but `peft` is not installed."
            ) from exc

        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        with open(adapter_config_path, "r") as file:
            adapter_conf = json.load(file)

        base_model_name_or_path = adapter_conf.get("base_model_name_or_path")
        if not base_model_name_or_path:
            raise ValueError(
                f"`base_model_name_or_path` missing in {adapter_config_path}"
            )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            torch_dtype=dtype,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, model_path).merge_and_unload()
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name_or_path, padding_side="left"
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = (
            tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        )
    model.eval()
    return model, tokenizer


def main():
    model, tokenizer = _load_model_and_tokenizer(LLM_PATH)

    inputs = tokenizer(PROMPT, return_tensors="pt")
    model_device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(model_device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model_device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            top_k=50,
            top_p=1.0,
            temperature=0.1,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(generated_text)


if __name__ == "__main__":
    main()
