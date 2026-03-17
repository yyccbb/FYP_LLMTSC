import argparse
from src.utils.utils import run_debugpy_server
from src.utils.config import DIC_CITY_ALIASES, DIC_CITY_SPECS, DIC_PATHS
from src.utils.errors import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj_name", type=str, default="LLMTSC_benchmark_all")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--city", type=str, default="all")
    parser.add_argument("--action_interval", type=int, default=30)
    parser.add_argument("--debug", type=bool, default=False)
    return parser.parse_args()

def 

def main(in_args):
    if in_args.debug:
        run_debugpy_server()
    
    if in_args.city.lower() != "all":
        city = DIC_CITY_ALIASES.get(in_args.city.lower())
        if city is None:
            raise InvalidCityError(city)
        

if __name__ == "__main__":
    args = parse_args()
    main(args)
