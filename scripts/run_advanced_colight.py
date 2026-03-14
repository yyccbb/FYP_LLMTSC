from src.utils.utils import pipeline_wrapper, merge, prepare_paths, run_debugpy_server
from src.utils.config import DIC_CITY_SPECS, DIC_CITY_ALIASES, DIC_BASE_AGENT_CONF, DIC_TRAFFIC_ENV_CONF
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj_name", type=str, default="LLMTSCS_critic_training")
    parser.add_argument("--model", type=str, default="AdvancedColight")
    parser.add_argument("--run_convenient_name", type=str, default="AdvancedColight")
    parser.add_argument("--city", type=str, default="hangzhou")
    parser.add_argument("--traffic_file", type=str, default="anon_4_4_hangzhou_real.json")
    parser.add_argument("--action_interval", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--gen", type=int, default=1)
    parser.add_argument("--debug", type=bool, default=False)
    return parser.parse_args()


def main(in_args=None):
    if in_args.debug:
        run_debugpy_server()

    city_dir_name = DIC_CITY_ALIASES.get(in_args.city.lower())
    if city_dir_name is None:
        raise ValueError(
            f"Unsupported city '{in_args.city}'. Available cities: {', '.join(DIC_CITY_SPECS.keys())}"
        )

    city_specs = DIC_CITY_SPECS[city_dir_name]
    count = city_specs.count
    road_net = city_specs.road_net
    traffic_file_list = city_specs.list_traffic_files
    num_rounds = in_args.epochs

    if in_args.traffic_file not in traffic_file_list:
        raise ValueError(
            f"Unsupported traffic_file '{in_args.traffic_file}' for city '{in_args.city}'. "
            f"Available files: {', '.join(traffic_file_list)}"
        )

    NUM_COL = int(road_net.split('_')[1])
    NUM_ROW = int(road_net.split('_')[0])
    num_intersections = NUM_ROW * NUM_COL
    print(in_args.traffic_file)
    print('num_intersections:', num_intersections)

    dic_agent_conf_extra = {
        "CNN_layers": [[32, 32]],
    }
    deploy_dic_agent_conf = merge(DIC_BASE_AGENT_CONF, dic_agent_conf_extra)

    dic_traffic_env_conf_extra = {
        "NUM_ROUNDS": num_rounds,
        "NUM_GENERATORS": in_args.gen,
        "NUM_AGENTS": 1,
        "NUM_INTERSECTIONS": num_intersections,
        "RUN_COUNTS": count,
        "MODEL_NAME": in_args.model,
        "MODEL": in_args.model,
        "PROJECT_NAME": in_args.proj_name,
        "NUM_ROW": NUM_ROW,
        "NUM_COL": NUM_COL,
        'MIN_ACTION_TIME': in_args.action_interval,
        'MEASURE_TIME': in_args.action_interval,
        "TRAFFIC_FILE": in_args.traffic_file,
        "ROADNET_FILE": f"roadnet_{road_net}.json",
        "LIST_STATE_FEATURE": [
            "cur_phase",
            "traffic_movement_pressure_queue_efficient",
            "lane_enter_running_part",
            "adjacency_matrix",
        ],

        "DIC_REWARD_INFO": {
            "queue_length": -0.25,
        },
    }
    deploy_dic_traffic_env_conf = merge(DIC_TRAFFIC_ENV_CONF, dic_traffic_env_conf_extra)

    dic_paths = prepare_paths(city_dir_name, in_args.run_convenient_name, in_args.traffic_file)

    pipeline_wrapper(dic_agent_conf=deploy_dic_agent_conf,
                     dic_traffic_env_conf=deploy_dic_traffic_env_conf,
                     dic_path=dic_paths,
                     roadnet=f'{city_dir_name}-{road_net}',
                     trafficflow=in_args.traffic_file.split(".")[0],
                     run_name=in_args.run_convenient_name)

    return in_args.run_convenient_name


if __name__ == "__main__":
    args = parse_args()

    main(args)
