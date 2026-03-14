"""
Colight agent implemented with PyTorch only.
observations: [lane_num_vehicle, cur_phase]
reward: -queue_length
"""
import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .agent import Agent


def build_memory():
    return []


class RepeatVector3D(nn.Module):
    def __init__(self, times):
        super().__init__()
        self.times = times

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.times, input_shape[1], input_shape[2]

    def forward(self, inputs):
        # [batch, agent, dim] -> [batch, 1, agent, dim] -> [batch, agent, agent, dim]
        return inputs.unsqueeze(1).repeat(1, self.times, 1, 1)

    def call(self, inputs):
        return self.forward(inputs)


class CoLightAttentionBlock(nn.Module):
    def __init__(self, num_agents, num_neighbors, d_in=128, h_dim=16, dout=128, head=8):
        super().__init__()
        self.num_agents = num_agents
        self.num_neighbors = num_neighbors
        self.d_in = d_in
        self.h_dim = h_dim
        self.dout = dout
        self.head = head

        self.repeat3d = RepeatVector3D(num_agents)
        self.agent_repr = nn.Linear(d_in, h_dim * head)
        self.neighbor_repr = nn.Linear(d_in, h_dim * head)
        self.neighbor_hidden_repr = nn.Linear(d_in, h_dim * head)
        self.mlp_after_relation = nn.Linear(h_dim, dout)

        self._reset_parameters()

    def _reset_parameters(self):
        for layer in [
            self.agent_repr,
            self.neighbor_repr,
            self.neighbor_hidden_repr,
            self.mlp_after_relation,
        ]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.05)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, in_feats, in_nei):
        """
        in_feats: [batch, agent, dim]
        in_nei: [batch, agent, neighbor, agent]
        """
        batch_size = in_feats.shape[0]

        # [batch, agent, dim] -> [batch, agent, 1, dim]
        agent_repr = in_feats.unsqueeze(2)

        # [batch, agent, dim] -> [batch, agent, agent, dim]
        neighbor_repr = self.repeat3d(in_feats)

        # [batch, agent, neighbor, agent] x [batch, agent, agent, dim] -> [batch, agent, neighbor, dim]
        neighbor_repr = torch.matmul(in_nei, neighbor_repr)

        # [batch, agent, 1, dim] -> [batch, agent, head, 1, h_dim]
        agent_repr_head = F.relu(self.agent_repr(agent_repr))
        agent_repr_head = agent_repr_head.reshape(
            batch_size, self.num_agents, 1, self.h_dim, self.head
        ).permute(0, 1, 4, 2, 3)

        # [batch, agent, neighbor, dim] -> [batch, agent, head, neighbor, h_dim]
        neighbor_repr_head = F.relu(self.neighbor_repr(neighbor_repr))
        neighbor_repr_head = neighbor_repr_head.reshape(
            batch_size, self.num_agents, self.num_neighbors, self.h_dim, self.head
        ).permute(0, 1, 4, 2, 3)

        # [batch, agent, head, 1, h_dim] x [batch, agent, head, h_dim, neighbor] -> [batch, agent, head, 1, neighbor]
        att = torch.matmul(agent_repr_head, neighbor_repr_head.transpose(-1, -2))
        att = F.softmax(att, dim=-1)
        att_record = att.reshape(batch_size, self.num_agents, self.head, self.num_neighbors)

        # self embedding again
        neighbor_hidden_repr_head = F.relu(self.neighbor_hidden_repr(neighbor_repr))
        neighbor_hidden_repr_head = neighbor_hidden_repr_head.reshape(
            batch_size, self.num_agents, self.num_neighbors, self.h_dim, self.head
        ).permute(0, 1, 4, 2, 3)

        out = torch.matmul(att, neighbor_hidden_repr_head).mean(dim=2)
        out = out.reshape(batch_size, self.num_agents, self.h_dim)
        out = F.relu(self.mlp_after_relation(out))
        return out, att_record


class CoLightQNetwork(nn.Module):
    def __init__(
        self,
        num_agents,
        num_neighbors,
        len_feature,
        num_actions,
        mlp_layers=None,
        cnn_layers=None,
    ):
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [32, 32]
        if cnn_layers is None:
            cnn_layers = [[32, 32]]

        self.num_agents = num_agents
        self.num_neighbors = num_neighbors
        self.len_feature = len_feature
        self.num_actions = num_actions

        self.embed_layers = nn.ModuleList()
        in_dim = len_feature
        for layer_size in mlp_layers:
            linear = nn.Linear(in_dim, layer_size)
            nn.init.normal_(linear.weight, mean=0.0, std=0.05)
            nn.init.zeros_(linear.bias)
            self.embed_layers.append(linear)
            in_dim = layer_size

        self.attention_layers = nn.ModuleList()
        cur_dim = in_dim
        for layer_size in cnn_layers:
            h_dim, out_dim = int(layer_size[0]), int(layer_size[1])
            self.attention_layers.append(
                CoLightAttentionBlock(
                    num_agents=num_agents,
                    num_neighbors=num_neighbors,
                    d_in=cur_dim,
                    h_dim=h_dim,
                    dout=out_dim,
                    head=5,
                )
            )
            cur_dim = out_dim

        self.action_layer = nn.Linear(cur_dim, num_actions)
        nn.init.normal_(self.action_layer.weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.action_layer.bias)

    def forward(self, inputs):
        feats, adjs = inputs
        if not torch.is_tensor(feats):
            feats = torch.tensor(feats, dtype=torch.float32)
        if not torch.is_tensor(adjs):
            adjs = torch.tensor(adjs, dtype=torch.float32)

        device = next(self.parameters()).device
        feats = feats.to(device=device, dtype=torch.float32)
        adjs = adjs.to(device=device, dtype=torch.float32)

        h = feats
        for layer in self.embed_layers:
            h = F.relu(layer(h))

        for layer in self.attention_layers:
            h, _ = layer(h, adjs)

        out = self.action_layer(h)
        return out


class CoLightAgentTorch(Agent):
    def __init__(
        self,
        dic_agent_conf=None,
        dic_traffic_env_conf=None,
        dic_path=None,
        cnt_round=None,
        intersection_id="0",
    ):
        super().__init__(dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.CNN_layers = dic_agent_conf["CNN_layers"]
        self.num_agents = dic_traffic_env_conf["NUM_INTERSECTIONS"]
        self.num_neighbors = min(dic_traffic_env_conf["TOP_K_ADJACENCY"], self.num_agents)
        self.num_actions = len(self.dic_traffic_env_conf["PHASE"])
        self.len_feature = self._cal_len_feature()
        self.memory = build_memory()
        self.loss_fn = nn.MSELoss()
        self.Xs = None
        self.Y = None

        if cnt_round is None:
            cnt_round = 0

        if cnt_round == 0:
            self.q_network = self.build_network()
            if os.path.isdir(self.dic_path["PATH_TO_TRAINED_CHECKPOINTS"]) and os.listdir(self.dic_path["PATH_TO_TRAINED_CHECKPOINTS"]):
                round_0_path = self._get_model_path(f"round_0_inter_{intersection_id}")
                if os.path.exists(round_0_path):
                    self.load_network(f"round_0_inter_{intersection_id}")
            self.q_network_bar = self.build_network_from_copy(self.q_network)
        else:
            try:
                self.load_network(f"round_{cnt_round - 1}_inter_{self.intersection_id}")
                if "UPDATE_Q_BAR_EVERY_C_ROUND" in self.dic_agent_conf:
                    if self.dic_agent_conf["UPDATE_Q_BAR_EVERY_C_ROUND"]:
                        bar_round = max(
                            (cnt_round - 1)
                            // self.dic_agent_conf["UPDATE_Q_BAR_FREQ"]
                            * self.dic_agent_conf["UPDATE_Q_BAR_FREQ"],
                            0,
                        )
                    else:
                        bar_round = max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0)
                else:
                    bar_round = max(cnt_round - self.dic_agent_conf["UPDATE_Q_BAR_FREQ"], 0)
                self.load_network_bar(f"round_{bar_round}_inter_{self.intersection_id}")
            except Exception:
                print(f"fail to load network, current round: {cnt_round}")
                self.q_network = self.build_network()
                self.q_network_bar = self.build_network_from_copy(self.q_network)

        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=self.dic_agent_conf["LEARNING_RATE"]
        )
        self.q_network.eval()
        self.q_network_bar.eval()

        decayed_epsilon = self.dic_agent_conf["EPSILON"] * pow(
            self.dic_agent_conf["EPSILON_DECAY"], cnt_round
        )
        self.dic_agent_conf["EPSILON"] = max(decayed_epsilon, self.dic_agent_conf["MIN_EPSILON"])

    def _cal_len_feature(self):
        n = 0
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:-1]
        for feat_name in used_feature:
            if "cur_phase" in feat_name:
                n += 8
            else:
                n += 12
        return n

    @staticmethod
    def MLP(ins, layers=None):
        """
        Kept for API compatibility.
        """
        if layers is None:
            layers = [128, 128]
        if not torch.is_tensor(ins):
            raise TypeError("MLP expects a torch.Tensor input")

        h = ins
        in_dim = h.shape[-1]
        for layer_size in layers:
            dense = nn.Linear(in_dim, layer_size).to(h.device)
            nn.init.normal_(dense.weight, mean=0.0, std=0.05)
            nn.init.zeros_(dense.bias)
            h = F.relu(dense(h))
            in_dim = layer_size
        return h

    def MultiHeadsAttModel(
        self, in_feats, in_nei, d_in=128, h_dim=16, dout=128, head=8, suffix=-1
    ):
        """
        Kept for API compatibility.
        """
        _ = suffix
        block = CoLightAttentionBlock(
            num_agents=self.num_agents,
            num_neighbors=self.num_neighbors,
            d_in=d_in,
            h_dim=h_dim,
            dout=dout,
            head=head,
        ).to(self.device)
        return block(in_feats.to(self.device), in_nei.to(self.device))

    def adjacency_index2matrix(self, adjacency_index):
        # [batch, agents, neighbors]
        adjacency_index_new = np.sort(adjacency_index, axis=-1).astype(np.int64)
        eye = np.eye(self.num_agents, dtype=np.float32)
        return eye[adjacency_index_new]

    def convert_state_to_input(self, s):
        """
        s: [state1, state2, ..., staten]
        """
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:-1]
        feats0 = []
        adj = []
        for i in range(self.num_agents):
            adj.append(s[i]["adjacency_matrix"])
            tmp = []
            for feature in used_feature:
                if feature == "cur_phase":
                    if self.dic_traffic_env_conf["BINARY_PHASE_EXPANSION"]:
                        tmp.extend(self.dic_traffic_env_conf["PHASE"][s[i][feature][0]])
                    else:
                        tmp.extend(s[i][feature])
                else:
                    tmp.extend(s[i][feature])
            feats0.append(tmp)

        feats = np.array([feats0], dtype=np.float32)
        adj = self.adjacency_index2matrix(np.array([adj], dtype=np.int64))
        return [feats, adj]

    def _predict_q(self, network, xs):
        network.eval()
        with torch.no_grad():
            q_values = network(xs).detach().cpu().numpy()
        return q_values

    def choose_action(self, count, states):
        """
        choose the best action for current state
        -input: state:[[state inter1],[state inter2], ...]
        -output: act: [#agents]
        """
        _ = count
        xs = self.convert_state_to_input(states)
        q_values = self._predict_q(self.q_network, xs)
        if random.random() <= self.dic_agent_conf["EPSILON"]:
            action = np.random.randint(self.num_actions, size=len(q_values[0]))
        else:
            action = np.argmax(q_values[0], axis=1)
        return action

    def choose_action_with_value(self, count, states):
        """
        choose the best action for current state
        -input: state:[[state inter1],[state inter2], ...]
        -output: act: [#agents], values: [#agents, #actions]
        """
        _ = count
        xs = self.convert_state_to_input(states)
        q_values = self._predict_q(self.q_network, xs)
        if random.random() <= self.dic_agent_conf["EPSILON"]:
            action = np.random.randint(self.num_actions, size=len(q_values[0]))
        else:
            action = np.argmax(q_values[0], axis=1)

        values = torch.softmax(torch.tensor(q_values[0], dtype=torch.float32) / 0.05, dim=1)
        v_min = values.min(dim=1, keepdim=True).values
        v_max = values.max(dim=1, keepdim=True).values
        norm_values = (values - v_min) / (v_max - v_min + 1e-8)
        return action, norm_values.numpy()

    @staticmethod
    def _concat_list(ls):
        tmp = []
        for i in range(len(ls)):
            tmp += ls[i]
        return [tmp]

    def prepare_Xs_Y(self, memory):
        """
        memory: [slice_data, slice_data, ..., slice_data]
        prepare memory for training
        """
        slice_size = len(memory[0])
        _adjs = []
        _state = [[] for _ in range(self.num_agents)]
        _next_state = [[] for _ in range(self.num_agents)]
        _action = [[] for _ in range(self.num_agents)]
        _reward = [[] for _ in range(self.num_agents)]
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"][:-1]

        for i in range(slice_size):
            _adj = []
            for j in range(self.num_agents):
                state, action, next_state, reward, _, _, _ = memory[j][i]
                _action[j].append(action)
                _reward[j].append(reward)
                _adj.append(state["adjacency_matrix"])
                _state[j].append(
                    self._concat_list([state[used_feature[k]] for k in range(len(used_feature))])
                )
                _next_state[j].append(
                    self._concat_list([next_state[used_feature[k]] for k in range(len(used_feature))])
                )
            _adjs.append(_adj)

        _adjs2 = self.adjacency_index2matrix(np.array(_adjs, dtype=np.int64))
        _state2 = np.concatenate([np.array(ss, dtype=np.float32) for ss in _state], axis=1)
        _next_state2 = np.concatenate([np.array(ss, dtype=np.float32) for ss in _next_state], axis=1)

        target = self._predict_q(self.q_network, [_state2, _adjs2])
        next_state_qvalues = self._predict_q(self.q_network_bar, [_next_state2, _adjs2])

        final_target = np.copy(target)
        for i in range(slice_size):
            for j in range(self.num_agents):
                final_target[i, j, _action[j][i]] = (
                    _reward[j][i] / self.dic_agent_conf["NORMAL_FACTOR"]
                    + self.dic_agent_conf["GAMMA"] * np.max(next_state_qvalues[i, j])
                )

        self.Xs = [_state2, _adjs2]
        self.Y = final_target.astype(np.float32)

    def build_network(self, MLP_layers=None):
        if MLP_layers is None:
            MLP_layers = [32, 32]
        network = CoLightQNetwork(
            num_agents=self.num_agents,
            num_neighbors=self.num_neighbors,
            len_feature=self.len_feature,
            num_actions=self.num_actions,
            mlp_layers=MLP_layers,
            cnn_layers=self.CNN_layers,
        ).to(self.device)
        return network

    def train_network(self):
        if self.Xs is None or self.Y is None or len(self.Y) == 0:
            return

        epochs = self.dic_agent_conf["EPOCHS"]
        batch_size = min(self.dic_agent_conf["BATCH_SIZE"], len(self.Y))
        patience = self.dic_agent_conf["PATIENCE"]

        feats = torch.tensor(self.Xs[0], dtype=torch.float32, device=self.device)
        adjs = torch.tensor(self.Xs[1], dtype=torch.float32, device=self.device)
        target = torch.tensor(self.Y, dtype=torch.float32, device=self.device)

        num_samples = feats.shape[0]
        split_idx = int(num_samples * 0.7)
        if split_idx <= 0:
            split_idx = num_samples

        best_state = None
        best_val = float("inf")
        patience_count = 0

        self.q_network.train()
        for _ in range(epochs):
            for start in range(0, split_idx, batch_size):
                end = min(start + batch_size, split_idx)
                pred = self.q_network([feats[start:end], adjs[start:end]])
                loss = self.loss_fn(pred, target[start:end])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if split_idx < num_samples:
                self.q_network.eval()
                with torch.no_grad():
                    val_pred = self.q_network([feats[split_idx:], adjs[split_idx:]])
                    val_loss = self.loss_fn(val_pred, target[split_idx:]).item()
                self.q_network.train()

                if val_loss < best_val:
                    best_val = val_loss
                    best_state = copy.deepcopy(self.q_network.state_dict())
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= patience:
                        if best_state is not None:
                            self.q_network.load_state_dict(best_state)
                        break

        self.q_network.eval()

    def build_network_from_copy(self, network_copy):
        network = copy.deepcopy(network_copy).to(self.device)
        network.load_state_dict(network_copy.state_dict())
        network.eval()
        return network

    def build_network_from_copy_only_weight(self, network, network_copy):
        network.load_state_dict(network_copy.state_dict())
        network.eval()
        return network

    def _get_model_path(self, file_name, file_path=None):
        if file_path is None:
            file_path = self.dic_path["PATH_TO_TRAINED_CHECKPOINTS"]
        return os.path.join(file_path, f"{file_name}.pt")

    def load_network(self, file_name, file_path=None):
        model_path = self._get_model_path(file_name, file_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {model_path}."
            )
        self.q_network = self.build_network()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint)
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=self.dic_agent_conf["LEARNING_RATE"]
        )
        self.q_network.eval()
        print(f"succeed in loading model {file_name}")

    def load_network_bar(self, file_name, file_path=None):
        model_path = self._get_model_path(file_name, file_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {model_path}."
            )
        self.q_network_bar = self.build_network()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.q_network_bar.load_state_dict(checkpoint)
        self.q_network_bar.eval()
        print(f"succeed in loading model {file_name}")

    def save_network(self, file_name):
        model_path = self._get_model_path(file_name)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.q_network.state_dict(), model_path)

    def save_network_bar(self, file_name):
        model_path = self._get_model_path(file_name)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.q_network_bar.state_dict(), model_path)
