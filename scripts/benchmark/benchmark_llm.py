import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.utils import run_debugpy_server
from src.utils.config import DIC_CITY_ALIASES, DIC_CITY_SPECS, DIC_PATHS, DIC_TRAFFIC_ENV_CONF
from src.utils.errors import InvalidCityError


METRIC_FIELDS = [
    "city",
    "traffic_file",
    "ATT",
    "AQL",
    "AWT",
    "run_counts",
    "action_interval",
    "steps",
    "checkpoint_path",
]

SIGNAL_PATTERN = re.compile(r"<signal>(.*?)</signal>", re.IGNORECASE | re.DOTALL)
DEFAULT_SIGNAL = "ETWT"
DEFAULT_VLLM_BATCH_SIZE = 16
DEFAULT_MAX_NEW_TOKENS = 1024

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./ft_models/merged/llama_ift_13b_jinan_1",
        help=(
            "LLM model path or LoRA adapter path. Kept as `checkpoint_path` to mirror "
            "benchmark_dqn.py flag naming."
        ),
    )
    parser.add_argument(
        "--city",
        type=str,
        default="all",
        help="all | hangzhou | jinan | newyork",
    )
    parser.add_argument("--action_interval", type=int, default=30)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default="benchmark_logs")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress simulator internal prints to keep logs clean (default: enabled).",
    )
    parser.add_argument(
        "--no-quiet",
        action="store_false",
        dest="quiet",
        help="Show simulator internal prints.",
    )
    parser.set_defaults(quiet=True)
    return parser.parse_args()


@contextmanager
def suppress_output(enabled=True):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield


def parse_roadnet_shape(road_net):
    num_row, num_col = road_net.split("_")
    return int(num_row), int(num_col)


def resolve_cities(city_flag):
    if city_flag.lower() == "all":
        return list(DIC_CITY_SPECS.keys())
    city = DIC_CITY_ALIASES.get(city_flag.lower())
    if city is None:
        raise InvalidCityError(city_flag)
    return [city]


def validate_action_interval(selected_cities, action_interval):
    for city_name in selected_cities:
        run_counts = DIC_CITY_SPECS[city_name].count
        if run_counts % action_interval != 0:
            raise ValueError(
                f"action_interval={action_interval} does not divide city count "
                f"{run_counts} for {city_name}."
            )


def _normalize_if_exists(path_str):
    expanded = os.path.expanduser(path_str)
    if os.path.exists(expanded):
        return os.path.normpath(expanded)
    return path_str


def _safe_name(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "adapter"


def _build_merge_key(adapter_path, adapter_conf):
    data = {
        "adapter_path": os.path.abspath(adapter_path),
        "adapter_conf": adapter_conf,
    }
    return hashlib.sha1(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()[:10]


def _is_lora_adapter_path(model_path):
    if not os.path.isdir(model_path):
        return False
    return os.path.isfile(os.path.join(model_path, "adapter_config.json"))


def _ensure_merged_model_cached(adapter_path, adapter_conf, merged_model_dir):
    merged_config = os.path.join(merged_model_dir, "config.json")
    if os.path.isfile(merged_config):
        return

    base_model_name_or_path = adapter_conf.get("base_model_name_or_path")
    if not base_model_name_or_path:
        raise ValueError(
            f"`base_model_name_or_path` missing in {os.path.join(adapter_path, 'adapter_config.json')}"
        )

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "LoRA merge requires `torch`, `transformers`, and `peft`. "
            "Install dependencies first."
        ) from exc

    os.makedirs(merged_model_dir, exist_ok=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(merged_model_dir)

    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    tokenizer.save_pretrained(merged_model_dir)

    del peft_model
    del base_model
    del merged_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_model_spec(checkpoint_path, output_dir):
    checkpoint_path_input = checkpoint_path
    normalized_input = _normalize_if_exists(checkpoint_path)

    spec = {
        "checkpoint_path_input": checkpoint_path_input,
        "normalized_input": normalized_input,
        "resolved_model_path": normalized_input,
        "is_lora_adapter": False,
        "base_model_name_or_path": None,
        "merged_model_path": None,
    }

    if isinstance(normalized_input, str) and os.path.exists(normalized_input):
        if os.path.isfile(normalized_input):
            return spec
        if not os.path.isdir(normalized_input):
            raise ValueError(f"checkpoint_path must be a file or directory. Got: {normalized_input}")

    if not isinstance(normalized_input, str) or not _is_lora_adapter_path(normalized_input):
        return spec

    adapter_config_path = os.path.join(normalized_input, "adapter_config.json")
    with open(adapter_config_path, "r") as f:
        adapter_conf = json.load(f)
    base_model_name_or_path = adapter_conf.get("base_model_name_or_path")
    if not base_model_name_or_path:
        raise ValueError(
            f"`base_model_name_or_path` missing in {adapter_config_path}"
        )

    merge_key = _build_merge_key(normalized_input, adapter_conf)
    adapter_name = _safe_name(os.path.basename(os.path.normpath(normalized_input)))
    merged_model_dir = os.path.join(
        os.path.normpath(output_dir),
        "merged_models",
        f"{adapter_name}_{merge_key}",
    )

    _ensure_merged_model_cached(
        adapter_path=normalized_input,
        adapter_conf=adapter_conf,
        merged_model_dir=merged_model_dir,
    )

    spec["resolved_model_path"] = merged_model_dir
    spec["is_lora_adapter"] = True
    spec["base_model_name_or_path"] = base_model_name_or_path
    spec["merged_model_path"] = merged_model_dir
    return spec


def init_vllm_runtime(resolved_model_path):
    try:
        import vllm
    except ImportError as exc:
        raise ImportError(
            "vLLM is required for this benchmark script. "
            "Please install vllm==0.11.0 in your runtime."
        ) from exc

    llm = vllm.LLM(
        model=resolved_model_path,
        tokenizer=resolved_model_path,
        dtype="bfloat16",
    )
    sampling_params = vllm.SamplingParams(
        top_k=50,
        top_p=1.0,
        temperature=0.1,
        max_tokens=2048 + DEFAULT_MAX_NEW_TOKENS,
    )
    return llm, sampling_params


def prepare_workdir(output_dir, city_name, traffic_file, roadnet_file):
    run_work_dir = os.path.join(
        output_dir, "cityflow_workdirs", city_name, traffic_file.replace(".json", "")
    )
    if os.path.exists(run_work_dir):
        shutil.rmtree(run_work_dir)
    os.makedirs(run_work_dir, exist_ok=True)

    city_data_dir = os.path.join(DIC_PATHS["PATH_TO_DATA"], city_name)
    shutil.copy2(os.path.join(city_data_dir, roadnet_file), os.path.join(run_work_dir, roadnet_file))
    shutil.copy2(os.path.join(city_data_dir, traffic_file), os.path.join(run_work_dir, traffic_file))
    return run_work_dir


def build_run_config(base_env_conf, road_net, traffic_file, action_interval, run_counts):
    num_row, num_col = parse_roadnet_shape(road_net)
    num_intersections = num_row * num_col

    env_conf = deepcopy(base_env_conf)
    env_conf.update(
        {
            "RUN_COUNTS": run_counts,
            "MIN_ACTION_TIME": action_interval,
            "MEASURE_TIME": action_interval,
            "TRAFFIC_FILE": traffic_file,
            "ROADNET_FILE": f"roadnet_{road_net}.json",
            "NUM_ROW": num_row,
            "NUM_COL": num_col,
            "NUM_INTERSECTIONS": num_intersections,
            "NUM_AGENTS": num_intersections,
            "MODEL_NAME": "LLM_VLLM_BENCHMARK",
            "MODEL": "LLM_VLLM_BENCHMARK",
            "PROJECT_NAME": "",
        }
    )
    return env_conf


def compute_metrics(env, run_counts):
    vehicle_travel_times = {}
    for inter in env.list_intersection:
        arrive_leave_times = inter.dic_vehicle_arrive_leave_time
        for vehicle_id, record in arrive_leave_times.items():
            if "shadow" in vehicle_id:
                continue
            enter_time = record["enter_time"]
            leave_time = record["leave_time"]
            if not np.isnan(enter_time):
                leave_time = leave_time if not np.isnan(leave_time) else run_counts
                vehicle_travel_times.setdefault(vehicle_id, []).append(leave_time - enter_time)

    avg_travel_time = (
        float(np.mean([sum(v) for v in vehicle_travel_times.values()]))
        if vehicle_travel_times
        else 0.0
    )
    return avg_travel_time


def build_llm_prompt(statistic_state, get_prompt_fn, state_to_text_fn):
    prompt_parts = get_prompt_fn(state_to_text_fn(statistic_state))
    return (
        prompt_parts[0]["content"]
        + "\n\n### Instruction:\n"
        + prompt_parts[1]["content"]
        + "\n\n### Response:\n"
    )


def generate_batched_responses(llm, sampling_params, prompts, batch_size=DEFAULT_VLLM_BATCH_SIZE):
    all_responses = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        responses_meta = llm.generate(prompts=batch_prompts, sampling_params=sampling_params)
        all_responses.extend(res.outputs[0].text for res in responses_meta)
    return all_responses


def decode_signal(response_text, valid_signal_to_code):
    matches = SIGNAL_PATTERN.findall(response_text)
    signal_text = matches[-1].strip().upper() if matches else DEFAULT_SIGNAL
    if signal_text not in valid_signal_to_code:
        signal_text = DEFAULT_SIGNAL
    return signal_text


def run_single_benchmark(
    llm,
    sampling_params,
    base_env_conf,
    city_name,
    road_net,
    traffic_file,
    output_dir,
    action_interval,
    run_counts,
    checkpoint_path_value,
    quiet=True,
):
    from src.env.cityflow_env import CityFlowEnv
    from src.utils.my_utils import action2code, four_phase_list, getPrompt, get_state_detail, state2text

    env_conf = build_run_config(
        base_env_conf, road_net, traffic_file, action_interval, run_counts
    )

    run_work_dir = prepare_workdir(output_dir, city_name, traffic_file, env_conf["ROADNET_FILE"])
    run_log_dir = os.path.join(output_dir, "sim_logs", city_name, traffic_file.replace(".json", ""))
    os.makedirs(run_log_dir, exist_ok=True)

    dic_path = {
        "PATH_TO_DATA": os.path.join(DIC_PATHS["PATH_TO_DATA"], city_name),
        "PATH_TO_TRAINED_CHECKPOINTS": "",
        "PATH_TO_WORK_DIRECTORY": run_work_dir,
    }

    try:
        with suppress_output(quiet):
            env = CityFlowEnv(
                path_to_log=run_log_dir,
                path_to_work_directory=run_work_dir,
                dic_traffic_env_conf=env_conf,
                dic_path=dic_path,
            )

            done = False
            state = env.reset()
            queue_length_episode = []
            waiting_time_episode = []

            total_steps = int(run_counts / action_interval)
            for _ in range(total_steps):
                if done:
                    break

                current_states = []
                for i in range(len(state)):
                    intersection = env.intersection_dict[env.list_intersection[i].inter_name]
                    roads = deepcopy(intersection["roads"])
                    statistic_state, _, _ = get_state_detail(roads, env)
                    current_states.append(statistic_state)

                prompts = [
                    build_llm_prompt(s, get_prompt_fn=getPrompt, state_to_text_fn=state2text)
                    for s in current_states
                ]
                responses = generate_batched_responses(
                    llm=llm,
                    sampling_params=sampling_params,
                    prompts=prompts,
                )

                action_list = []
                for response in responses:
                    signal_text = decode_signal(response, valid_signal_to_code=four_phase_list)
                    action_list.append(action2code(signal_text))

                next_state, _, done, _ = env.step(action_list)
                state = next_state

                queue_length_inter = [
                    sum(inter.dic_feature["lane_num_waiting_vehicle_in"]) for inter in env.list_intersection
                ]
                queue_length_episode.append(sum(queue_length_inter))

                waiting_times = [v["time"] for v in env.waiting_vehicle_list.values()]
                waiting_time_episode.append(float(np.mean(waiting_times)) if waiting_times else 0.0)

            avg_travel_time = compute_metrics(env, run_counts)
            env.end_cityflow()
    finally:
        if quiet:
            shutil.rmtree(run_work_dir, ignore_errors=True)
            shutil.rmtree(run_log_dir, ignore_errors=True)

    avg_queue_length = float(np.mean(queue_length_episode)) if queue_length_episode else 0.0
    avg_waiting_time = float(np.mean(waiting_time_episode)) if waiting_time_episode else 0.0

    return {
        "city": city_name,
        "traffic_file": traffic_file,
        "ATT": avg_travel_time,
        "AQL": avg_queue_length,
        "AWT": avg_waiting_time,
        "run_counts": run_counts,
        "action_interval": action_interval,
        "steps": int(run_counts / action_interval),
        "checkpoint_path": checkpoint_path_value,
    }


def init_metrics_logger(output_dir, model_spec, city_flag):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"benchmark_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    csv_path = os.path.join(run_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDS)
        writer.writeheader()

    metadata = {
        "timestamp": timestamp,
        "checkpoint_path_input": model_spec["checkpoint_path_input"],
        "checkpoint_path_normalized": model_spec["normalized_input"],
        "resolved_model_path": model_spec["resolved_model_path"],
        "is_lora_adapter": model_spec["is_lora_adapter"],
        "base_model_name_or_path": model_spec["base_model_name_or_path"],
        "merged_model_path": model_spec["merged_model_path"],
        "city_flag": city_flag,
    }
    with open(os.path.join(run_dir, "metadata.json"), "w") as file:
        json.dump(metadata, file, indent=2)

    return run_dir, csv_path


def append_metric(csv_path, row):
    with open(csv_path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDS)
        writer.writerow(row)


def main(in_args):
    if in_args.debug:
        run_debugpy_server()

    output_dir = os.path.normpath(os.path.expanduser(in_args.output_dir))
    selected_cities = resolve_cities(in_args.city)
    validate_action_interval(selected_cities, in_args.action_interval)

    model_spec = resolve_model_spec(in_args.checkpoint_path, output_dir)
    run_dir, csv_path = init_metrics_logger(
        output_dir=output_dir,
        model_spec=model_spec,
        city_flag=in_args.city,
    )

    llm, sampling_params = init_vllm_runtime(model_spec["resolved_model_path"])

    base_env_conf = deepcopy(DIC_TRAFFIC_ENV_CONF)
    total_jobs = sum(len(DIC_CITY_SPECS[city_name].list_traffic_files) for city_name in selected_cities)
    completed = 0

    for city_name in selected_cities:
        city_specs = DIC_CITY_SPECS[city_name]
        run_counts = city_specs.count
        traffic_files = city_specs.list_traffic_files

        for traffic_file in traffic_files:
            completed += 1
            row = run_single_benchmark(
                llm=llm,
                sampling_params=sampling_params,
                base_env_conf=base_env_conf,
                city_name=city_name,
                road_net=city_specs.road_net,
                traffic_file=traffic_file,
                output_dir=run_dir,
                action_interval=in_args.action_interval,
                run_counts=run_counts,
                checkpoint_path_value=in_args.checkpoint_path,
                quiet=in_args.quiet,
            )
            append_metric(csv_path, row)

            print(
                f"[{completed}/{total_jobs}] {row['city']} | {row['traffic_file']} "
                f"| ATT={row['ATT']:.4f} AQL={row['AQL']:.4f} AWT={row['AWT']:.4f}"
            )

    print(f"Saved benchmark logs to: {run_dir}")
    print(f"Per-file metrics: {os.path.join(run_dir, 'metrics.csv')}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
