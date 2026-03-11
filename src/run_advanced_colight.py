from src.utils.utils import pipeline_wrapper, merge, prepare_paths, run_debugpy_server
from src.utils import config
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

    traffic_file_list = []

    if in_args.city == 'jinan':
        count = 3600
        road_net = "3_4"
        traffic_file_list = ["anon_3_4_jinan_real.json", "anon_3_4_jinan_real_2000.json",
                             "anon_3_4_jinan_real_2500.json", "anon_3_4_jinan_synthetic_24000_60min.json"]
        num_rounds = in_args.epochs
        city_dir_name = "Jinan"
    elif in_args.city == 'hangzhou':
        count = 3600
        road_net = "4_4"
        traffic_file_list = ["anon_4_4_hangzhou_real.json", "anon_4_4_hangzhou_real_5816.json", "anon_4_4_hangzhou_synthetic_24000_60min.json"]
        num_rounds = in_args.epochs
        city_dir_name = "Hangzhou"
    elif in_args.city == 'newyork':
        count = 3600
        road_net = "28_7"
        traffic_file_list = ["anon_28_7_newyork_real_double.json", "anon_28_7_newyork_real_triple.json"]
        num_rounds = in_args.epochs
        city_dir_name = "NewYork"

    assert in_args.traffic_file in traffic_file_list

    NUM_COL = int(road_net.split('_')[1])
    NUM_ROW = int(road_net.split('_')[0])
    num_intersections = NUM_ROW * NUM_COL
    print(in_args.traffic_file)
    print('num_intersections:', num_intersections)

    dic_agent_conf_extra = {
        "CNN_layers": [[32, 32]],
    }
    deploy_dic_agent_conf = merge(getattr(config, "DIC_BASE_AGENT_CONF"), dic_agent_conf_extra)

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
        "ROADNET_FILE": "roadnet_{0}.json".format(road_net),
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
    deploy_dic_traffic_env_conf = merge(config.dic_traffic_env_conf, dic_traffic_env_conf_extra)

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
