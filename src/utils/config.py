from dataclasses import dataclass

@dataclass(frozen=True)
class CitySpecs:
    count: int
    road_net: str
    list_traffic_files: tuple[str, ...]

DIC_CITY_SPECS = {
    "Jinan": CitySpecs(
        3600, 
        "3_4", 
        (
            "anon_3_4_jinan_real.json",
            "anon_3_4_jinan_real_2000.json",
            "anon_3_4_jinan_real_2500.json",
            "anon_3_4_jinan_synthetic_24000_60min.json"
        )
    ),
    "Hangzhou": CitySpecs(
        3600,
        "4_4",
        (
            "anon_4_4_hangzhou_real.json", 
            "anon_4_4_hangzhou_real_5816.json",
            "anon_4_4_hangzhou_synthetic_24000_60min.json"
        )
    ),
    "NewYork": CitySpecs(
        3600,
        "28_7",
        (
            "anon_28_7_newyork_real_double.json",
            "anon_28_7_newyork_real_triple.json"
        )
    )
}

DIC_CITY_ALIASES = {city_name.lower(): city_name for city_name in DIC_CITY_SPECS.keys()}

DIC_PATH = {
    "PATH_TO_TRAINED_CHECKPOINTS": "checkpoints/default",
    "PATH_TO_WORK_DIRECTORY": "records/default",
    "PATH_TO_DATA": "data/template",
    "PATH_TO_ERROR": "errors/default",
}

DIC_TRAFFIC_ENV_CONF = {

    "LIST_MODEL": ["Random", "Fixedtime",  "MaxPressure", "EfficientMaxPressure", "AdvancedMaxPressure",
                   "EfficientPressLight", "EfficientColight", "EfficientMPLight",
                   "AdvancedMPLight", "AdvancedColight", "AdvancedDQN", "Attend"],
    "LIST_MODEL_NEED_TO_UPDATE": ["EfficientPressLight", "EfficientColight", "EfficientMPLight",
                                  "AdvancedMPLight", "AdvancedColight", "AdvancedDQN", "Attend"],

    "NUM_LANE": 12,
    # 'WT_ET', 'NT_ST', 'WL_EL', 'NL_SL'/ 'WL_WT', 'EL_ET', 'SL_ST', 'NL_NT'
   "FORGET_ROUND": 20,
    "RUN_COUNTS": 3600,
    "MODEL_NAME": None,
    "TOP_K_ADJACENCY": 5,

    "ACTION_PATTERN": "set",
    "NUM_INTERSECTIONS": 1,

    "OBS_LENGTH": 167,
    "MIN_ACTION_TIME": 30,
    "MEASURE_TIME": 30,

    "BINARY_PHASE_EXPANSION": True,

    "YELLOW_TIME": 5,
    "ALL_RED_TIME": 0,
    "NUM_PHASES": 4,
    "NUM_LANES": [3, 3, 3, 3],

    "INTERVAL": 1,

    "LIST_STATE_FEATURE": [
        "cur_phase",
        "time_this_phase",
        "lane_num_vehicle",
        "lane_num_vehicle_downstream",
        "traffic_movement_pressure_num",
        "traffic_movement_pressure_queue",
        "traffic_movement_pressure_queue_efficient",
        "pressure",
        "adjacency_matrix"
    ],
    "DIC_REWARD_INFO": {
        "queue_length": 0,
        "pressure": 0,
    },
    "PHASE": {
        1: [0, 1, 0, 1, 0, 0, 0, 0],
        2: [0, 0, 0, 0, 0, 1, 0, 1],
        3: [1, 0, 1, 0, 0, 0, 0, 0],
        4: [0, 0, 0, 0, 1, 0, 1, 0]
    },
    "list_lane_order": ["WL", "WT", "EL", "ET", "NL", "NT", "SL", "ST"],
    "PHASE_LIST": ['WT_ET', 'NT_ST', 'WL_EL', 'NL_SL'],

}

DIC_BASE_AGENT_CONF = {
    "D_DENSE": 20,
    "LEARNING_RATE": 0.001,
    "PATIENCE": 10,
    "BATCH_SIZE": 20,
    "EPOCHS": 100,
    "SAMPLE_SIZE": 3000,
    "MAX_MEMORY_LEN": 12000,

    "UPDATE_Q_BAR_FREQ": 5,
    "UPDATE_Q_BAR_EVERY_C_ROUND": False,

    "GAMMA": 0.8,
    "NORMAL_FACTOR": 20,

    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "LOSS_FUNCTION": "mean_squared_error",
}

DIC_CHATGPT_AGENT_CONF = {
    "GPT_VERSION": "gpt-4",
    "LOG_DIR": "../GPT_logs"
}

DIC_FIXEDTIME_AGENT_CONF = {
    "FIXED_TIME": [30, 30, 30, 30]
}

DIC_MAXPRESSURE_AGENT_CONF = {
    "FIXED_TIME": [30, 30, 30, 30]
}