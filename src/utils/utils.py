import wandb
import copy
import debugpy
import numpy as np
import time
import os

from src.utils.pipeline import Pipeline

location_dict_short = {"North": "N", "South": "S", "East": "E", "West": "W"}
location_direction_dict = ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]

def run_debugpy_server(port=None, wait_for_client=True):
    def _env_int(name):
        value = os.getenv(name)
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    local_rank = _env_int("LOCAL_RANK")
    if local_rank is None:
        local_rank = _env_int("RANK") or 0
    world_size = _env_int("WORLD_SIZE") or 1

    if port is None:
        base_port = _env_int("DEBUGPY_PORT")
        if base_port is None:
            launcher_port = _env_int("MAIN_PORT")
            if launcher_port is None:
                launcher_port = _env_int("MASTER_PORT")
            if launcher_port is not None:
                # Keep debugpy away from torch/accelerate's rendezvous port.
                base_port = launcher_port + 100
            else:
                base_port = 5678
    else:
        try:
            base_port = int(port)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid debugpy port: {port!r}") from exc

    debug_port = base_port + local_rank if world_size > 1 else base_port

    debugpy.listen(("0.0.0.0", debug_port))
    print(f"[rank {local_rank}] Debugger listening on {os.uname()[1]}:{debug_port}.")

    if world_size > 1 and local_rank != 0:
        print(f"[rank {local_rank}] Skipping wait_for_client in multi-process mode.")
        return debug_port

    if wait_for_client:
        print(f"[rank {local_rank}] Waiting for debugger attach...")
        debugpy.wait_for_client()

    return debug_port


def merge(dic_tmp, dic_to_change):
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)
    return dic_result

def prepare_paths(city_dir_name, run_convenient_name, traffic_file):
    dic_paths = {
        "PATH_TO_TRAINED_CHECKPOINTS": os.path.join("checkpoints", run_convenient_name, traffic_file + "_" +
                                      time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
        "PATH_TO_WORK_DIRECTORY": os.path.join("logs", run_convenient_name, traffic_file + "_"
                                               + time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
        "PATH_TO_DATA": os.path.join("data", city_dir_name),
        "PATH_TO_ERROR": os.path.join("errors", run_convenient_name)
    }
    return dic_paths

def pipeline_wrapper(dic_agent_conf, dic_traffic_env_conf, dic_path, roadnet, trafficflow, run_name):
    results_table = []
    all_rewards = []
    all_queue_len = []
    all_travel_time = []
    dic_path["PATH_TO_TRAINED_CHECKPOINTS"] = (dic_path["PATH_TO_TRAINED_CHECKPOINTS"].split(".")[0] + ".json" +
                                    time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time())))
    dic_path["PATH_TO_WORK_DIRECTORY"] = (dic_path["PATH_TO_WORK_DIRECTORY"].split(".")[0] + ".json" +
                                            time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time())))
    ppl = Pipeline(dic_agent_conf=dic_agent_conf,
                    dic_traffic_env_conf=dic_traffic_env_conf,
                    dic_path=dic_path,
                    roadnet=roadnet,
                    trafficflow=trafficflow)
    round_results = ppl.run(run_name, multi_process=False)
    results_table.append([round_results['test_reward_over'], round_results['test_avg_queue_len_over'], round_results['test_avg_travel_time_over']])
    all_rewards.append(round_results['test_reward_over'])
    all_queue_len.append(round_results['test_avg_queue_len_over'])
    all_travel_time.append(round_results['test_avg_travel_time_over'])

    # delete junk
    # cmd_delete_model = 'find <dir> -type f ! -name "round_<round>_inter_*.h5" -exec rm -rf {} \;'.replace("<dir>", dic_path["PATH_TO_TRAINED_CHECKPOINTS"]).replace("<round>", str(int(dic_traffic_env_conf["NUM_ROUNDS"] - 1)))
    # cmd_delete_work = 'find <dir> -type f ! -name "state_action.json" -exec rm -rf {} \;'.replace("<dir>", dic_path["PATH_TO_WORK_DIRECTORY"])
    # os.system(cmd_delete_model)
    # os.system(cmd_delete_work)

    # results_table.append([np.average(all_rewards), np.average(all_queue_len), np.average(all_travel_time)])
    # results_table.append([np.std(all_rewards), np.std(all_queue_len), np.std(all_travel_time)])

    table_logger = wandb.init(
        project=dic_traffic_env_conf['PROJECT_NAME'],
        group=f"{dic_traffic_env_conf['MODEL']}-{roadnet}-{trafficflow}-{len(dic_traffic_env_conf['PHASE'])}_Phases",
        name="exp_results",
        config=merge(merge(dic_agent_conf, dic_path), dic_traffic_env_conf),
    )
    columns = ["reward", "avg_queue_len", "avg_travel_time"]
    logger_table = wandb.Table(columns=columns, data=results_table)
    table_logger.log({"results": logger_table})
    wandb.finish()

    print("pipeline_wrapper end")
    return
