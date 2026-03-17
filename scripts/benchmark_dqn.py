import argparse
import csv
import json
import os
import shutil
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime

import numpy as np

# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import DIC_CITY_ALIASES, DIC_CITY_SPECS, DIC_PATHS
from src.utils.errors import InvalidCityError
from src.utils.utils import run_debugpy_server


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
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


def resolve_checkpoint_spec(checkpoint_path):
    ckpt_path = os.path.expanduser(checkpoint_path)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    if not os.path.isfile(ckpt_path):
        raise ValueError(
            f"checkpoint_path must be a checkpoint file, not a directory: {ckpt_path}"
        )

    return {
        "path": os.path.normpath(ckpt_path),
    }


def load_benchmark_configs(checkpoint_path):
    checkpoint_dir = os.path.normpath(os.path.dirname(checkpoint_path))
    sep = os.sep
    rel_prefix = f"checkpoints{sep}"
    abs_segment = f"{sep}checkpoints{sep}"

    if checkpoint_dir == "checkpoints":
        config_dir = "logs"
    elif checkpoint_dir.startswith(rel_prefix):
        config_dir = os.path.join("logs", checkpoint_dir[len(rel_prefix) :])
    elif abs_segment in checkpoint_dir:
        config_dir = checkpoint_dir.replace(abs_segment, f"{sep}logs{sep}", 1)
    else:
        raise ValueError(
            "checkpoint_path must be under a 'checkpoints' root directory so config "
            f"can be mapped to logs. Got: {checkpoint_path}"
        )

    with open(os.path.join(config_dir, "agent.conf"), "r") as file:
        dic_agent_conf = json.load(file)

    anon_env_conf_path = os.path.join(config_dir, "anon_env.conf")
    traffic_env_conf_path = os.path.join(config_dir, "traffic_env.conf")

    if os.path.exists(anon_env_conf_path):
        with open(anon_env_conf_path, "r") as file:
            dic_traffic_env_conf = json.load(file)
    elif os.path.exists(traffic_env_conf_path):
        with open(traffic_env_conf_path, "r") as file:
            dic_traffic_env_conf = json.load(file)
    else:
        raise FileNotFoundError(
            f"Could not find anon_env.conf or traffic_env.conf in {config_dir}"
        )
    
    phase = dic_traffic_env_conf.get("PHASE")
    if isinstance(phase, dict):
        dic_traffic_env_conf["PHASE"] = {int(k): v for k, v in phase.items()}

    return dic_agent_conf, dic_traffic_env_conf


def load_agent_checkpoint(agent, checkpoint_file):
    import torch

    try:
        state_dict = torch.load(checkpoint_file, map_location=agent.device, weights_only=True)
    except TypeError:
        # Backward compatibility for older PyTorch versions without `weights_only`.
        state_dict = torch.load(checkpoint_file, map_location=agent.device)
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint must be a raw state_dict dictionary.")

    agent.q_network = agent.build_network()
    agent.q_network.load_state_dict(state_dict)
    agent.optimizer = torch.optim.Adam(
        agent.q_network.parameters(), lr=agent.dic_agent_conf["LEARNING_RATE"]
    )
    agent.q_network.eval()


def load_checkpoints_for_agents(agents, checkpoint_spec):
    checkpoint_file = checkpoint_spec["path"]
    for agent in agents:
        load_agent_checkpoint(agent, checkpoint_file)


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


def build_run_config(base_env_conf, city_name, road_net, traffic_file, action_interval, run_counts):
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
            "NUM_AGENTS": env_conf.get("NUM_AGENTS", 1),
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


def run_single_benchmark(
    checkpoint_spec,
    base_agent_conf,
    base_env_conf,
    city_name,
    road_net,
    traffic_file,
    output_dir,
    action_interval,
    run_counts,
    num_agents,
    quiet=True,
):
    from src.env.cityflow_env import CityFlowEnv
    from src.modelling.agent.colight_agent import CoLightAgentTorch

    env_conf = build_run_config(
        base_env_conf, city_name, road_net, traffic_file, action_interval, run_counts
    )
    agent_conf = deepcopy(base_agent_conf)
    env_conf["NUM_AGENTS"] = num_agents

    if env_conf["MODEL"] in env_conf.get("LIST_MODEL_NEED_TO_UPDATE", []):
        agent_conf["EPSILON"] = 0
        agent_conf["MIN_EPSILON"] = 0

    run_work_dir = prepare_workdir(output_dir, city_name, traffic_file, env_conf["ROADNET_FILE"])
    run_log_dir = os.path.join(output_dir, "sim_logs", city_name, traffic_file.replace(".json", ""))
    os.makedirs(run_log_dir, exist_ok=True)
    checkpoint_dir = os.path.dirname(checkpoint_spec["path"])

    dic_path = {
        "PATH_TO_DATA": os.path.join(DIC_PATHS["PATH_TO_DATA"], city_name),
        "PATH_TO_TRAINED_CHECKPOINTS": checkpoint_dir,
        "PATH_TO_WORK_DIRECTORY": run_work_dir,
    }

    try:
        with suppress_output(quiet):
            agents = []
            for i in range(env_conf["NUM_AGENTS"]):
                agent = CoLightAgentTorch(
                    dic_agent_conf=agent_conf,
                    dic_traffic_env_conf=env_conf,
                    dic_path=dic_path,
                    cnt_round=0,
                    intersection_id=str(i),
                )
                agents.append(agent)

            load_checkpoints_for_agents(agents, checkpoint_spec)

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
            for step_num in range(total_steps):
                if done:
                    break

                action_list = []
                for i in range(env_conf["NUM_AGENTS"]):
                    action_list = agents[i].choose_action(step_num, state)

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
        "checkpoint_path": checkpoint_spec["path"],
    }


def init_metrics_logger(output_dir, checkpoint_path, city_flag):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"benchmark_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    csv_path = os.path.join(run_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=METRIC_FIELDS,
        )
        writer.writeheader()

    metadata = {
        "timestamp": timestamp,
        "checkpoint_path_input": checkpoint_path,
        "city_flag": city_flag,
    }
    with open(os.path.join(run_dir, "metadata.json"), "w") as file:
        json.dump(metadata, file, indent=2)

    return run_dir, csv_path


def append_metric(csv_path, row):
    with open(csv_path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDS)
        writer.writerow(row)


def resolve_cities(city_flag):
    if city_flag.lower() == "all":
        return list(DIC_CITY_SPECS.keys())
    city = DIC_CITY_ALIASES.get(city_flag.lower())
    if city is None:
        raise InvalidCityError(city_flag)
    return [city]


def main(in_args):
    if in_args.debug:
        run_debugpy_server()

    checkpoint_spec = resolve_checkpoint_spec(in_args.checkpoint_path)
    base_agent_conf, base_env_conf = load_benchmark_configs(checkpoint_spec["path"])
    checkpoint_num_agents = int(base_env_conf.get("NUM_AGENTS", 1))
    # if checkpoint_num_agents < 1:
    #     checkpoint_num_agents = 1

    output_dir = os.path.normpath(in_args.output_dir)
    run_dir, csv_path = init_metrics_logger(
        output_dir=output_dir,
        checkpoint_path=in_args.checkpoint_path,
        city_flag=in_args.city,
    )

    selected_cities = resolve_cities(in_args.city)
    rows = []
    total_jobs = sum(len(DIC_CITY_SPECS[city_name].list_traffic_files) for city_name in selected_cities)
    completed = 0

    for city_name in selected_cities:
        city_specs = DIC_CITY_SPECS[city_name]
        run_counts = city_specs.count
        if run_counts % in_args.action_interval != 0:
            raise ValueError(
                f"action_interval={in_args.action_interval} does not divide city count "
                f"{run_counts} for {city_name}."
            )
        traffic_files = city_specs.list_traffic_files

        for traffic_file in traffic_files:
            completed += 1
            row = run_single_benchmark(
                checkpoint_spec=checkpoint_spec,
                base_agent_conf=base_agent_conf,
                base_env_conf=base_env_conf,
                city_name=city_name,
                road_net=city_specs.road_net,
                traffic_file=traffic_file,
                output_dir=run_dir,
                action_interval=in_args.action_interval,
                run_counts=run_counts,
                num_agents=checkpoint_num_agents,
                quiet=in_args.quiet,
            )
            append_metric(csv_path, row)
            rows.append(row)

            print(
                f"[{completed}/{total_jobs}] {row['city']} | {row['traffic_file']} "
                f"| ATT={row['ATT']:.4f} AQL={row['AQL']:.4f} AWT={row['AWT']:.4f}"
            )

    print(f"Saved benchmark logs to: {run_dir}")
    print(f"Per-file metrics: {os.path.join(run_dir, 'metrics.csv')}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
