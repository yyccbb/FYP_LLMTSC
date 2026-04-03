import pickle
import numpy as np
import json
import sys
import pandas as pd
import os
import copy
import cityflow as engine
import time
from multiprocessing import Process
from functools import reduce

from src.env.intersection import Intersection

location_dict = {"North": "N", "South": "S", "East": "E", "West": "W"}
location_dict_reverse = {"N": "North", "S": "South", "E": "East", "W": "West"}
direction_dict = {"go_straight": "T", "turn_left": "L", "turn_right": "R"}

def calculate_road_length(road_points):
    length = 0.0
    i = 1
    while i < len(road_points):
        length += np.sqrt((road_points[i]['x'] - road_points[i-1]['x']) ** 2 + (road_points[i]['y'] - road_points[i-1]['y']) ** 2)
        i += 1
    return length


class CityFlowEnv:
    def __init__(self, path_to_log, path_to_work_directory, dic_traffic_env_conf, dic_path):
        self.path_to_log = path_to_log
        self.path_to_work_directory = path_to_work_directory
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path

        self.current_time = None
        self.id_to_index = None
        self.traffic_light_node_dict = None
        self.intersection_dict = None
        self.eng = None
        self.list_intersection = None
        self.list_inter_log = None
        self.list_lanes = None
        self.system_states = None
        self.lane_length = None
        self.waiting_vehicle_list = {}

        # check min action time
        if self.dic_traffic_env_conf["MIN_ACTION_TIME"] <= self.dic_traffic_env_conf["YELLOW_TIME"]:
            """ include the yellow time in action time """
            print("MIN_ACTION_TIME should include YELLOW_TIME")
            sys.exit()

        # touch new inter_{}.pkl (if exists, remove)
        for inter_ind in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]):
            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            f.close()

    def reset(self):
        print(" ============= self.eng.reset() to be implemented ==========")
        cityflow_config = {
            "interval": self.dic_traffic_env_conf["INTERVAL"],
            "seed": int(np.random.randint(0, 100)),
            "laneChange": True,
            "dir": self.path_to_work_directory+"/",
            "roadnetFile": self.dic_traffic_env_conf["ROADNET_FILE"],
            "flowFile": self.dic_traffic_env_conf["TRAFFIC_FILE"],
            "rlTrafficLight": True,
            "saveReplay": bool(self.dic_traffic_env_conf.get("SAVE_REPLAY", True)),
            "roadnetLogFile": f"./{self.dic_traffic_env_conf['ROADNET_FILE']}-{self.dic_traffic_env_conf['TRAFFIC_FILE']}-{self.dic_traffic_env_conf['MODEL']}-{len(self.dic_traffic_env_conf['PHASE'])}_Phases-roadnetLogFile.json",
            "replayLogFile": f"./{self.dic_traffic_env_conf['ROADNET_FILE']}-{self.dic_traffic_env_conf['TRAFFIC_FILE']}-{self.dic_traffic_env_conf['MODEL']}-{len(self.dic_traffic_env_conf['PHASE'])}_Phases-replayLogFile.txt"
        }
        # print(cityflow_config)
        with open(os.path.join(self.path_to_work_directory, "cityflow.config"), "w") as json_file:
            json.dump(cityflow_config, json_file)

        self.eng = engine.Engine(os.path.join(self.path_to_work_directory, "cityflow.config"), thread_num=1)

        # get adjacency
        self.traffic_light_node_dict = self._adjacency_extraction()

        # get lane length
        _, self.lane_length = self.get_lane_length()

        # initialize intersections (grid)
        self.list_intersection = [Intersection((i+1, j+1), self.dic_traffic_env_conf, self.eng,
                                               self.traffic_light_node_dict["intersection_{0}_{1}".format(i+1, j+1)],
                                               self.path_to_log,
                                               self.lane_length)
                                  for i in range(self.dic_traffic_env_conf["NUM_COL"])
                                  for j in range(self.dic_traffic_env_conf["NUM_ROW"])]
        self.list_inter_log = [[] for _ in range(self.dic_traffic_env_conf["NUM_COL"] *
                                                 self.dic_traffic_env_conf["NUM_ROW"])]

        self.id_to_index = {}
        count = 0
        for i in range(self.dic_traffic_env_conf["NUM_COL"]):
            for j in range(self.dic_traffic_env_conf["NUM_ROW"]):
                self.id_to_index["intersection_{0}_{1}".format(i+1, j+1)] = count
                count += 1

        self.list_lanes = []
        for inter in self.list_intersection:
            self.list_lanes += inter.list_lanes
        self.list_lanes = np.unique(self.list_lanes).tolist()

        # get new measurements
        self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                              "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                              "get_vehicle_speed": self.eng.get_vehicle_speed(),
                              "get_vehicle_distance": self.eng.get_vehicle_distance(),
                              }

        for inter in self.list_intersection:
            inter.update_current_measurements(self.system_states)
        state, done = self.get_state()

        # create roadnet dict
        if self.intersection_dict is None:
            self.create_intersection_dict()

        return state


    def create_intersection_dict(self):
        roadnet_filepath = f"./{self.dic_path['PATH_TO_DATA']}/{self.dic_traffic_env_conf['ROADNET_FILE']}"
        try:
            with open(roadnet_filepath, "r") as file:
                roadnet = json.load(file)
        except Exception as e:
            raise e

        intersections_raw = roadnet["intersections"]
        roads_raw = roadnet["roads"]

        agent_intersections = {}

        # init agent intersections
        for i, inter in enumerate(intersections_raw):
            inter_id = inter["id"]
            intersection = None
            for env_inter in self.list_intersection:
                if env_inter.inter_name == inter_id:
                    intersection = env_inter
                    break

            if len(inter['roadLinks']) > 0:
                # collect yellow allowed road links
                yellow_time = None
                phases = inter['trafficLight']['lightphases']
                all_sets = []
                yellow_phase_idx = None
                for p_i, p in enumerate(phases):
                    all_sets.append(set(p['availableRoadLinks']))
                    if p["time"] < 30:
                        yellow_phase_idx = p_i
                        yellow_time = p["time"]
                yellow_allowed_links = reduce(lambda x, y: x & y, all_sets)

                # init intersection
                agent_intersections[inter_id] = {"phases": {"Y": {"time": yellow_time, "idx": yellow_phase_idx}},
                                                 "roads": {}}

                # init roads
                roads = {}
                for r in inter["roads"]:
                    roads[r] = {"location": None, "type": "incoming", "go_straight": None, "turn_left": None,
                                "turn_right": None, "length": None, "max_speed": None,
                                "lanes": {"go_straight": [], "turn_left": [], "turn_right": []}}

                # collect road length speed info & init road location
                road_links = inter["roadLinks"]
                for r in roads_raw:
                    r_id = r["id"]
                    if r_id in roads:
                        roads[r_id]["length"] = calculate_road_length(r["points"])
                        roads[r_id]["max_speed"] = r["lanes"][0]["maxSpeed"]
                        for env_road_location in intersection.dic_entering_approach_to_edge:
                            if intersection.dic_entering_approach_to_edge[env_road_location] == r_id:
                                roads[r_id]["location"] = location_dict_reverse[env_road_location]
                                break
                        for env_road_location in intersection.dic_exiting_approach_to_edge:
                            if intersection.dic_exiting_approach_to_edge[env_road_location] == r_id:
                                roads[r_id]["location"] = location_dict_reverse[env_road_location]
                                break

                # collect signal phase info
                for p_idx, p in enumerate(phases):
                    other_allowed_links = set(p['availableRoadLinks']) - yellow_allowed_links
                    if len(other_allowed_links) > 0:
                        allowed_directions = []
                        for l_idx in other_allowed_links:
                            link = road_links[l_idx]
                            location = roads[link["startRoad"]]["location"]
                            direction = link["type"]
                            allowed_directions.append(f"{location_dict[location]}{direction_dict[direction]}")
                        allowed_directions = sorted(allowed_directions)
                        allowed_directions = f"{allowed_directions[0]}{allowed_directions[1]}"
                        agent_intersections[inter_id]["phases"][allowed_directions] = {"time": p["time"], "idx": p_idx}

                # collect location type direction info
                for r_link in road_links:
                    start = r_link['startRoad']
                    end = r_link['endRoad']
                    lane_links = r_link['laneLinks']

                    for r in roads:
                        if r != start:
                            continue
                        # collect type
                        roads[r]["type"] = "outgoing"

                        # collect directions
                        if r_link["type"] == "go_straight":
                            roads[r]["go_straight"] = end

                            # collect lane info
                            for l_link in lane_links:
                                lane_id = l_link['startLaneIndex']
                                if lane_id not in roads[r]["lanes"]["go_straight"]:
                                    roads[r]["lanes"]["go_straight"].append(lane_id)

                        elif r_link["type"] == "turn_left":
                            roads[r]["turn_left"] = end

                            # collect lane info
                            for l_link in lane_links:
                                lane_id = l_link['startLaneIndex']
                                if lane_id not in roads[r]["lanes"]["turn_left"]:
                                    roads[r]["lanes"]["turn_left"].append(lane_id)

                        elif r_link["type"] == "turn_right":
                            roads[r]["turn_right"] = end

                            # collect lane info
                            for l_link in lane_links:
                                lane_id = l_link['startLaneIndex']
                                if lane_id not in roads[r]["lanes"]["turn_right"]:
                                    roads[r]["lanes"]["turn_right"].append(lane_id)

                agent_intersections[inter_id]["roads"] = roads

        self.intersection_dict = agent_intersections

    def step(self, action):

        step_start_time = time.time()

        list_action_in_sec = [action]
        list_action_in_sec_display = [action]
        for i in range(self.dic_traffic_env_conf["MIN_ACTION_TIME"]-1):
            if self.dic_traffic_env_conf["ACTION_PATTERN"] == "switch":
                list_action_in_sec.append(np.zeros_like(action).tolist())
            elif self.dic_traffic_env_conf["ACTION_PATTERN"] == "set":
                list_action_in_sec.append(np.copy(action).tolist())
            list_action_in_sec_display.append(np.full_like(action, fill_value=-1).tolist())

        average_reward_action_list = [0]*len(action)
        for i in range(self.dic_traffic_env_conf["MIN_ACTION_TIME"]):

            action_in_sec = list_action_in_sec[i]
            action_in_sec_display = list_action_in_sec_display[i]

            instant_time = self.get_current_time()
            self.current_time = self.get_current_time()

            before_action_feature = self.get_feature()
            # state = self.get_state()

            if i == 0:
                print("time: {0}".format(instant_time))
                    
            self._inner_step(action_in_sec)

            # get reward
            reward = self.get_reward()
            for j in range(len(reward)):
                average_reward_action_list[j] = (average_reward_action_list[j] * i + reward[j]) / (i + 1)
            self.log(cur_time=instant_time, before_action_feature=before_action_feature, action=action_in_sec_display)
            next_state, done = self.get_state()

        print("Step time: ", time.time() - step_start_time)
        return next_state, reward, done, average_reward_action_list

    def _inner_step(self, action):
        # copy current measurements to previous measurements
        for inter in self.list_intersection:
            inter.update_previous_measurements()
        # set signals
        # multi_intersection decided by action {inter_id: phase}
        for inter_ind, inter in enumerate(self.list_intersection):
            inter.set_signal(
                action=action[inter_ind],
                action_pattern=self.dic_traffic_env_conf["ACTION_PATTERN"],
                yellow_time=self.dic_traffic_env_conf["YELLOW_TIME"],
                path_to_log=self.path_to_log
            )

        # run one step
        for i in range(int(1/self.dic_traffic_env_conf["INTERVAL"])):
            self.eng.next_step()

            # update queuing vehicle info
            vehicle_ids = self.eng.get_vehicles(include_waiting=False)
            for v_id in vehicle_ids:
                v_info = self.eng.get_vehicle_info(v_id)
                speed = float(v_info["speed"])
                if speed < 0.1:
                    if v_id not in self.waiting_vehicle_list:
                        self.waiting_vehicle_list[v_id] = {"time": None, "link": None}
                        self.waiting_vehicle_list[v_id]["time"] = self.dic_traffic_env_conf["INTERVAL"]
                        self.waiting_vehicle_list[v_id]["link"] = v_info['drivable']
                    else:
                        if self.waiting_vehicle_list[v_id]["link"] != v_info['drivable']:
                            self.waiting_vehicle_list[v_id] = {"time": None, "link": None}
                            self.waiting_vehicle_list[v_id]["time"] = self.dic_traffic_env_conf["INTERVAL"]
                            self.waiting_vehicle_list[v_id]["link"] = v_info['drivable']
                        else:
                            self.waiting_vehicle_list[v_id]["time"] += self.dic_traffic_env_conf["INTERVAL"]
                else:
                    if v_id in self.waiting_vehicle_list:
                        self.waiting_vehicle_list.pop(v_id)

                if v_id in self.waiting_vehicle_list and self.waiting_vehicle_list[v_id]["link"] != v_info['drivable']:
                    self.waiting_vehicle_list.pop(v_id)

        self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                              "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                              "get_vehicle_speed": self.eng.get_vehicle_speed(),
                              "get_vehicle_distance": self.eng.get_vehicle_distance()
                              }

        for inter in self.list_intersection:
            inter.update_current_measurements(self.system_states)

    def get_feature(self):
        list_feature = [inter.get_feature() for inter in self.list_intersection]
        return list_feature

    def get_state(self, list_state_feature=None):
        if list_state_feature is not None:
            list_state = [inter.get_state(list_state_feature) for inter in self.list_intersection]
            done = False
        else:
            list_state = [inter.get_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"]) for inter in self.list_intersection]
            done = False
        return list_state, done

    def get_reward(self):
        list_reward = [inter.get_reward(self.dic_traffic_env_conf["DIC_REWARD_INFO"]) for inter in self.list_intersection]
        return list_reward

    def get_current_time(self):
        return self.eng.get_current_time()

    @staticmethod
    def _intersection_snapshot_fields():
        return [
            "current_phase_index",
            "previous_phase_index",
            "next_phase_to_set_index",
            "current_phase_duration",
            "all_red_flag",
            "all_yellow_flag",
            "flicker",
            "dic_lane_vehicle_previous_step",
            "dic_lane_vehicle_previous_step_in",
            "dic_lane_waiting_vehicle_count_previous_step",
            "dic_vehicle_speed_previous_step",
            "dic_vehicle_distance_previous_step",
            "dic_lane_vehicle_current_step_in",
            "dic_lane_vehicle_current_step",
            "dic_lane_waiting_vehicle_count_current_step",
            "dic_vehicle_speed_current_step",
            "dic_vehicle_distance_current_step",
            "list_lane_vehicle_previous_step_in",
            "list_lane_vehicle_current_step_in",
            "dic_vehicle_arrive_leave_time",
            "dic_feature",
            "dic_feature_previous_step",
        ]

    def _capture_intersection_snapshot(self, inter):
        snapshot = {}
        for field_name in self._intersection_snapshot_fields():
            snapshot[field_name] = copy.deepcopy(getattr(inter, field_name))
        return snapshot

    def _load_intersection_snapshot(self, inter, inter_snapshot):
        for field_name in self._intersection_snapshot_fields():
            if field_name not in inter_snapshot:
                raise KeyError(
                    f"Intersection snapshot for {inter.inter_name} is missing '{field_name}'."
                )
            setattr(inter, field_name, copy.deepcopy(inter_snapshot[field_name]))

    def capture_snapshot(self):
        """
        Capture an in-memory snapshot of both CityFlow engine state and
        Python-side environment/intersection state.
        """
        if self.eng is None or self.list_intersection is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
        if not hasattr(self.eng, "snapshot"):
            raise NotImplementedError("CityFlow Engine does not expose snapshot().")

        snapshot = {
            "engine_snapshot": self.eng.snapshot(),
            "current_time": self.current_time,
            "waiting_vehicle_list": copy.deepcopy(self.waiting_vehicle_list),
            "system_states": copy.deepcopy(self.system_states),
            "intersection_states": {},
        }
        for inter in self.list_intersection:
            snapshot["intersection_states"][inter.inter_name] = self._capture_intersection_snapshot(inter)

        return snapshot

    def load_snapshot(self, snapshot):
        """
        Restore an in-memory snapshot previously created by capture_snapshot().
        """
        if self.eng is None or self.list_intersection is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
        if snapshot is None:
            raise ValueError("snapshot cannot be None.")
        if "engine_snapshot" not in snapshot:
            raise KeyError("snapshot is missing 'engine_snapshot'.")
        if not hasattr(self.eng, "load"):
            raise NotImplementedError("CityFlow Engine does not expose load().")

        self.eng.load(snapshot["engine_snapshot"])
        self.current_time = self.get_current_time()
        self.waiting_vehicle_list = copy.deepcopy(snapshot.get("waiting_vehicle_list", {}))
        self.system_states = copy.deepcopy(snapshot.get("system_states", {}))

        intersection_states = snapshot.get("intersection_states")
        if intersection_states is None:
            raise KeyError("snapshot is missing 'intersection_states'.")

        missing_intersections = [
            inter.inter_name for inter in self.list_intersection
            if inter.inter_name not in intersection_states
        ]
        if missing_intersections:
            raise KeyError(
                f"snapshot is missing states for intersections: {missing_intersections}"
            )

        for inter in self.list_intersection:
            self._load_intersection_snapshot(inter, intersection_states[inter.inter_name])

    def log(self, cur_time, before_action_feature, action):

        for inter_ind in range(len(self.list_intersection)):
            self.list_inter_log[inter_ind].append({"time": cur_time,
                                                   "state": before_action_feature[inter_ind],
                                                   "action": action[inter_ind]})

    def batch_log_2(self):
        """
        Used for model test, only log the vehicle_inter_.csv
        """
        for inter_ind in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]):
            # changed from origin
            if int(inter_ind) % 100 == 0:
                print("Batch log for inter ", inter_ind)
            path_to_log_file = os.path.join(self.path_to_log, "vehicle_inter_{0}.csv".format(inter_ind))
            dic_vehicle = self.list_intersection[inter_ind].get_dic_vehicle_arrive_leave_time()
            df = pd.DataFrame.from_dict(dic_vehicle, orient="index")
            df.to_csv(path_to_log_file, na_rep="nan")

    def batch_log(self, start, stop):
        for inter_ind in range(start, stop):
            # changed from origin
            if int(inter_ind) % 100 == 0:
                print("Batch log for inter ", inter_ind)
            path_to_log_file = os.path.join(self.path_to_log, "vehicle_inter_{0}.csv".format(inter_ind))
            dic_vehicle = self.list_intersection[inter_ind].get_dic_vehicle_arrive_leave_time()
            df = pd.DataFrame.from_dict(dic_vehicle, orient="index")
            df.to_csv(path_to_log_file, na_rep="nan")
            
            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            pickle.dump(self.list_inter_log[inter_ind], f)
            f.close()

    def bulk_log_multi_process(self, batch_size=100):
        assert len(self.list_intersection) == len(self.list_inter_log)
        if batch_size > len(self.list_intersection):
            batch_size_run = len(self.list_intersection)
        else:
            batch_size_run = batch_size
        process_list = []
        for batch in range(0, len(self.list_intersection), batch_size_run):
            start = batch
            stop = min(batch + batch_size, len(self.list_intersection))
            p = Process(target=self.batch_log, args=(start, stop))
            print("before")
            p.start()
            print("end")
            process_list.append(p)
        print("before join")

        for t in process_list:
            t.join()
        print("end join")

    def _adjacency_extraction(self):
        traffic_light_node_dict = {}
        file = os.path.join(self.path_to_work_directory, self.dic_traffic_env_conf["ROADNET_FILE"])
        with open("{0}".format(file)) as json_data:
            net = json.load(json_data)
            for inter in net["intersections"]:
                if not inter["virtual"]:
                    traffic_light_node_dict[inter["id"]] = {"location": {"x": float(inter["point"]["x"]),
                                                                         "y": float(inter["point"]["y"])},
                                                            "total_inter_num": None, "adjacency_row": None,
                                                            "inter_id_to_index": None,
                                                            "neighbor_ENWS": None}

            top_k = self.dic_traffic_env_conf["TOP_K_ADJACENCY"]
            total_inter_num = len(traffic_light_node_dict.keys())
            inter_id_to_index = {}

            edge_id_dict = {}
            for road in net["roads"]:
                if road["id"] not in edge_id_dict.keys():
                    edge_id_dict[road["id"]] = {}
                edge_id_dict[road["id"]]["from"] = road["startIntersection"]
                edge_id_dict[road["id"]]["to"] = road["endIntersection"]

            index = 0
            for i in traffic_light_node_dict.keys():
                inter_id_to_index[i] = index
                index += 1

            for i in traffic_light_node_dict.keys():
                location_1 = traffic_light_node_dict[i]["location"]

                row = np.array([0]*total_inter_num)
                # row = np.zeros((self.dic_traffic_env_conf["NUM_ROW"],self.dic_traffic_env_conf["NUM_col"]))
                for j in traffic_light_node_dict.keys():
                    location_2 = traffic_light_node_dict[j]["location"]
                    dist = self._cal_distance(location_1, location_2)
                    row[inter_id_to_index[j]] = dist
                if len(row) == top_k:
                    adjacency_row_unsorted = np.argpartition(row, -1)[:top_k].tolist()
                elif len(row) > top_k:
                    adjacency_row_unsorted = np.argpartition(row, top_k)[:top_k].tolist()
                else:
                    adjacency_row_unsorted = [k for k in range(total_inter_num)]
                adjacency_row_unsorted.remove(inter_id_to_index[i])
                traffic_light_node_dict[i]["adjacency_row"] = [inter_id_to_index[i]]+adjacency_row_unsorted
                traffic_light_node_dict[i]["total_inter_num"] = total_inter_num

            for i in traffic_light_node_dict.keys():
                traffic_light_node_dict[i]["total_inter_num"] = inter_id_to_index
                traffic_light_node_dict[i]["neighbor_ENWS"] = []
                for j in range(4):
                    road_id = i.replace("intersection", "road")+"_"+str(j)
                    if edge_id_dict[road_id]["to"] not in traffic_light_node_dict.keys():
                        traffic_light_node_dict[i]["neighbor_ENWS"].append(None)
                    else:
                        traffic_light_node_dict[i]["neighbor_ENWS"].append(edge_id_dict[road_id]["to"])

        return traffic_light_node_dict

    @staticmethod
    def _cal_distance(loc_dict1, loc_dict2):
        a = np.array((loc_dict1["x"], loc_dict1["y"]))
        b = np.array((loc_dict2["x"], loc_dict2["y"]))
        return np.sqrt(np.sum((a-b)**2))

    @staticmethod
    def end_cityflow():
        print("============== cityflow process end ===============")

    def get_lane_length(self):
        """
        newly added part for get lane length
        Read the road net file
        Return: dict{lanes} normalized with the min lane length
        """
        file = os.path.join(self.path_to_work_directory, self.dic_traffic_env_conf["ROADNET_FILE"])
        with open(file) as json_data:
            net = json.load(json_data)
        roads = net['roads']
        lanes_length_dict = {}
        lane_normalize_factor = {}

        for road in roads:
            points = road["points"]
            road_length = abs(points[0]['x'] + points[0]['y'] - points[1]['x'] - points[1]['y'])
            for i in range(3):
                lane_id = road['id'] + "_{0}".format(i)
                lanes_length_dict[lane_id] = road_length
        min_length = min(lanes_length_dict.values())

        for key, value in lanes_length_dict.items():
            lane_normalize_factor[key] = value / min_length
        return lane_normalize_factor, lanes_length_dict
