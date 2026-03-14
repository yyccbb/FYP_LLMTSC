import argparse
import os
import time

from src.utils.llm_aft_trainer import LLM_CGPR_Collector
from src.utils.config import DIC_CITY_SPECS, DIC_TRAFFIC_ENV_CONF, DIC_CITY_ALIASES
from src.utils.utils import merge, run_debugpy_server

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memo", type=str, default='LLMTLCSIFTCollection')
    parser.add_argument("--llm_model", type=str, default="llama_ift_13b_jinan_1")
    parser.add_argument("--llm_path", type=str, default="./ft_models/merged/llama_ift_13b_jinan_1")
    parser.add_argument("--new_max_tokens", type=int, default=1024)
    parser.add_argument("--multi_process", action="store_true", default=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="jinan")
    parser.add_argument("--traffic_file", type=str, default="anon_3_4_jinan_real.json")
    parser.add_argument("--debug", type=bool, default=False)

    return parser.parse_args()


def main(in_args):
    if in_args.debug:
        run_debugpy_server()

    city_name = DIC_CITY_ALIASES.get(in_args.dataset.lower())
    if city_name is not None:
        city_specs = DIC_CITY_SPECS[city_name]
        count = city_specs.count
        road_net = city_specs.road_net
        traffic_file_list = city_specs.list_traffic_files
        template = city_name
    elif in_args.dataset == "template":
        count = 3600
        road_net = "1_1"
        traffic_file_list = ("flow_main_stream.json",)
        template = "template"
    else:
        raise ValueError(
            f"Unsupported dataset '{in_args.dataset}'. "
            f"Available datasets: {', '.join(sorted(set(DATASET_TO_CITY) | {'newyork_16x3', 'template'}))}"
        )
    in_args.model = in_args.memo

    if in_args.traffic_file not in traffic_file_list:
        raise ValueError(
            f"Unsupported traffic_file '{in_args.traffic_file}' for dataset '{in_args.dataset}'. "
            f"Available files: {', '.join(traffic_file_list)}"
        )

    NUM_ROW = int(road_net.split('_')[0])
    NUM_COL = int(road_net.split('_')[1])
    num_intersections = NUM_ROW * NUM_COL
    print('num_intersections:', num_intersections)
    print(in_args.traffic_file)

    dic_agent_conf_extra = {
        "LLM_PATH": in_args.llm_path,
        "LLM_MODEL": in_args.llm_model,
        "LOG_DIR": f"./{in_args.llm_model}_logs",
        "NEW_MAX_TOKENS": in_args.new_max_tokens
    }

    dic_traffic_env_conf_extra = {
        "NUM_AGENTS": num_intersections,
        "NUM_INTERSECTIONS": num_intersections,

        "MODEL_NAME": f"{in_args.model}-{dic_agent_conf_extra['LLM_MODEL']}",
        "PROJECT_NAME": "",
        "RUN_COUNTS": count,
        "NUM_ROW": NUM_ROW,
        "NUM_COL": NUM_COL,

        "TRAFFIC_FILE": in_args.traffic_file,
        "ROADNET_FILE": "roadnet_{0}.json".format(road_net),

        "LIST_STATE_FEATURE": [
            "cur_phase",
            "traffic_movement_pressure_queue",
        ],

        "DIC_REWARD_INFO": {
            'queue_length': -0.25
        }
    }

    dic_agent_conf_extra["FIXED_TIME"] = [30, 30, 30, 30]

    dic_traffic_env_conf_extra["NUM_AGENTS"] = dic_traffic_env_conf_extra["NUM_INTERSECTIONS"]
    dic_path_extra = {
        "PATH_TO_TRAINED_CHECKPOINTS": os.path.join("model", in_args.memo, in_args.traffic_file + "_" +
                                      time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
        "PATH_TO_WORK_DIRECTORY": os.path.join("records", in_args.memo, in_args.traffic_file + "_" +
                                               time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))),
        "PATH_TO_DATA": os.path.join("data", template, str(road_net))
    }

    trainer = LLM_CGPR_Collector(dic_agent_conf_extra,
                                 merge(DIC_TRAFFIC_ENV_CONF, dic_traffic_env_conf_extra),
                                 dic_path_extra,
                                 f'{template}-{road_net}', in_args.traffic_file.split(".")[0])

    trainer.train_test()

if __name__ == "__main__":
    args = parse_args()
    main(args)
