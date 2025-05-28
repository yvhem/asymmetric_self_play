import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

# Helper for converting dict observations
def process_obs(obs_dict):
    obs_tensors = [
        torch.tensor(obs_dict[key], dtype=torch.float32, device=device).flatten()
        for key in sorted(obs_dict.keys())
    ]
    return torch.cat(obs_tensors, dim=-1)

# Each object is embedded and max-pooled over the object dimension to produce a single vector
class PermutationInvariantEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim=256, output_dim=512):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, output_dim)
        )

    def forward(self, x):
        x = self.embed(x)
        x, _ = torch.max(x, dim=0, keepdim=True) # (1, output_dim) Max pooling over objects
        return x

# Only feed
# - Robot arm position, gripper position, object state, goal state.
# Object state
# - object position, rotation, velocity, rotational velocity, distance object-gripper, contact object-gripper
# Goal state
# - each object's desired position and rotation, relative distance between the current object state and the desired state
# In hybrid policy additionally feed 3 camera images
# - current state captured by a fixed camera in front of the table
# - current state captured from a camera mounted on the gripper wrist
# - goal state captured from the fixed camera
# Both Alice and Bob take robot and object state observations as input
# Alice does nto take goal state inputs

# First branch
# 1. Robot Joint Position: obs['robot_joint_pos'] (6,)
# 2. Embedding (256)
# 3. LayerNorm
# Second branch
# 1. Gripper Position: obs['gripper_pos'] (3,)
# 2. Embedding (256)
# 3. LayerNorm
# Third branch
# 1.1 Object State (both Bob and Alice):
#    - Object Position: obs['object_pos'] (num_objects, 3)
#    - Object Rotation: obs['object_rot'] (num_objects, 3)
#    - Object Velocity: obs['object_vel_pos'] (num_objects, 3)
#    - Object Rotational Velocity: obs['object_vel_rot'] (num_objects, 3)
#    - Distance Object-Gripper: obs['obj_rel_pos'] (num_objects, 3)
#    - Contact Object-Gripper: obs['obj_gripper_contact'] (num_objects, 2)
# 1.2 Goal State (only Bob):
#    - Goal Object Position: obs['goal_obj_pos'] (num_objects, 3)
#    - Goal Object Rotation: obs['goal_obj_rot'] (num_objects, 3)
#    - Distance current-desired position: obs['rel_goal_obj_pos'] (num_objects, 3)
#    - Distance current-desired rotation: obs['rel_goal_obj_rot'] (num_objects, 3)
# 2. Permutation Invariant Embedding (512)
# 3. LayerNorm
# Fourth branch (Hybrid policy only)
# 1. Vision Inputs
# 2. IMPALA
# 3. LayerNorm
# Concatenate all branches
# Sum block
# ReLU block
# MLP block
# LSTM block
# Action or Value Head block
class Policy(nn.Module):
    def __init__(self, action_dims, is_goal_conditioned=False, lstm_hidden_size=512):
        super().__init__()
        self.action_dims = action_dims
        self.is_goal_conditioned = is_goal_conditioned
        self.hidden_size = lstm_hidden_size

        # Robot joint position
        self.robot_fc = nn.Sequential(
            nn.Linear(6, 256),  # obs['robot_joint_pos'] (6,)
            nn.LayerNorm(256),
            nn.ReLU()
        )

        # Gripper position
        self.gripper_fc = nn.Sequential(
            nn.Linear(3, 256),  # obs['gripper_pos'] (3,)
            nn.LayerNorm(256),
            nn.ReLU()
        )

        # Object state + Goal state (Bob only)
        object_state_dim = 3+3+3+3+3+2  # object_pos, object_rot, object_vel_pos, object_vel_rot, obj_rel_pos, obj_gripper_contact
        goal_state_dim = 3+3+3+3        # goal_obj_pos, goal_obj_rot, rel_goal_obj_pos, rel_goal_obj_rot
        total_obj_dim = object_state_dim + (goal_state_dim if is_goal_conditioned else 0)

        # Permutation-invariant object embedding
        self.object_embedding = PermutationInvariantEmbedding(input_dim=total_obj_dim, output_dim=512)
        self.object_layer_norm = nn.LayerNorm(512)

        self.projection = nn.Linear(256 + 256 + 512, self.hidden_size) # 1024 -> 512 

        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)

        # Residual MLP block
        self.fc_residual = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.fc_residual_relu = nn.ReLU()

        # Separate heads for action logits and value
        self.policy_head = nn.Linear(self.hidden_size, sum(action_dims))
        self.value_head = nn.Linear(self.hidden_size, 1)

    # get features before LSTM for a single obs_dict
    def _get_common_features(self, obs_dict):
        x_robot = self.robot_fc(torch.tensor(obs_dict['robot_joint_pos'], dtype=torch.float32, device=device)).unsqueeze(0)
        x_gripper = self.gripper_fc(torch.tensor(obs_dict['gripper_pos'], dtype=torch.float32, device=device)).unsqueeze(0)

        obj_pos_tensor = torch.tensor(obs_dict['obj_pos'], dtype=torch.float32, device=device)
        if obj_pos_tensor.ndim == 1: obj_pos_tensor = obj_pos_tensor.unsqueeze(0)

        obj_rot_tensor = torch.tensor(obs_dict['obj_rot'], dtype=torch.float32, device=device)
        if obj_rot_tensor.ndim == 1: obj_rot_tensor = obj_rot_tensor.unsqueeze(0)

        obj_vel_pos_tensor = torch.tensor(obs_dict['obj_vel_pos'], dtype=torch.float32, device=device)
        if obj_vel_pos_tensor.ndim == 1: obj_vel_pos_tensor = obj_vel_pos_tensor.unsqueeze(0)

        obj_vel_rot_tensor = torch.tensor(obs_dict['obj_vel_rot'], dtype=torch.float32, device=device)
        if obj_vel_rot_tensor.ndim == 1: obj_vel_rot_tensor = obj_vel_rot_tensor.unsqueeze(0)

        obj_rel_pos_tensor = torch.tensor(obs_dict['obj_rel_pos'], dtype=torch.float32, device=device)
        if obj_rel_pos_tensor.ndim == 1: obj_rel_pos_tensor = obj_rel_pos_tensor.unsqueeze(0)

        obj_gripper_contact_tensor = torch.tensor(obs_dict['obj_gripper_contact'], dtype=torch.float32, device=device)
        if obj_gripper_contact_tensor.ndim == 1: obj_gripper_contact_tensor = obj_gripper_contact_tensor.unsqueeze(0)

        obj_states_list = [
            obj_pos_tensor, obj_rot_tensor, obj_vel_pos_tensor,
            obj_vel_rot_tensor, obj_rel_pos_tensor, obj_gripper_contact_tensor
        ]

        if self.is_goal_conditioned:
            goal_obj_pos_tensor = torch.tensor(obs_dict['goal_obj_pos'], dtype=torch.float32, device=device)
            if goal_obj_pos_tensor.ndim == 1: goal_obj_pos_tensor = goal_obj_pos_tensor.unsqueeze(0)
            
            goal_obj_rot_tensor = torch.tensor(obs_dict['goal_obj_rot'], dtype=torch.float32, device=device)
            if goal_obj_rot_tensor.ndim == 1: goal_obj_rot_tensor = goal_obj_rot_tensor.unsqueeze(0)

            rel_goal_obj_pos_tensor = torch.tensor(obs_dict['rel_goal_obj_pos'], dtype=torch.float32, device=device)
            if rel_goal_obj_pos_tensor.ndim == 1: rel_goal_obj_pos_tensor = rel_goal_obj_pos_tensor.unsqueeze(0)

            rel_goal_obj_rot_tensor = torch.tensor(obs_dict['rel_goal_obj_rot'], dtype=torch.float32, device=device)
            if rel_goal_obj_rot_tensor.ndim == 1: rel_goal_obj_rot_tensor = rel_goal_obj_rot_tensor.unsqueeze(0)
            
            goal_states_list = [
                goal_obj_pos_tensor, goal_obj_rot_tensor,
                rel_goal_obj_pos_tensor, rel_goal_obj_rot_tensor
            ]
            obj_states_list.extend(goal_states_list)
        
        # PermutationInvariantEmbedding expects (num_objects, features)
        obj_states_combined = torch.cat(obj_states_list, dim=-1)
        x_obj_embedded = self.object_embedding(obj_states_combined) # output (1, 512)
        x_obj = self.object_layer_norm(x_obj_embedded)

        # concat features for LSTM input: x_robot, x_gripper, x_obj (all (1, D))
        common_features = torch.cat([x_robot, x_gripper, x_obj], dim=-1) # (1, 1024)
        projected_features = self.projection(common_features)            # (1, 512)
        return projected_features

    def forward(self, obs_sequence, hx_initial, cx_initial):
        all_logits, all_values = [], []
        hx = torch.zeros(1, self.hidden_size, device=device) if hx_initial is None else hx_initial
        cx = torch.zeros(1, self.hidden_size, device=device) if cx_initial is None else cx_initial
        
        for obs_dict in obs_sequence:
            projected_features = self._get_common_features(obs_dict) # (1, self.hidden_size)
            hx, cx = self.lstm(projected_features, (hx, cx))         # LSTM step

            # residual MLP block
            residual_input = hx
            x_after_residual = self.fc_residual(residual_input)
            x_after_residual = self.fc_residual_relu(residual_input + x_after_residual)

            # heads
            logits = self.policy_head(x_after_residual) # (1, sum_action_dims)
            value = self.value_head(x_after_residual)   # (1, 1)

            all_logits.append(logits)
            all_values.append(value)
        
        return torch.cat(all_logits, dim=0), torch.cat(all_values, dim=0), hx, cx

    def get_action(self, obs_dict, hx=None, cx=None):
        projected_features = self._get_common_features(obs_dict)
        if hx is None: hx = torch.zeros(1, self.hidden_size, device=device)
        if cx is None: cx = torch.zeros(1, self.hidden_size, device=device)

        # LSTM step
        hx_new, cx_new = self.lstm(projected_features, (hx, cx))

        # residual MLP block
        residual_input = hx_new
        x_after_residual = self.fc_residual(residual_input)
        x_after_residual = self.fc_residual_relu(residual_input + x_after_residual)

        logits = self.policy_head(x_after_residual)

        # action sampling
        split_logits = torch.split(logits.squeeze(0), self.action_dims, dim=-1)
        action_dists = [Categorical(logits=lg) for lg in split_logits]
        actions = [dist.sample().item() for dist in action_dists]

        log_probs_list = []
        for i, dist in enumerate(action_dists):
            action_tensor = torch.tensor(actions[i], device=device, dtype=torch.long)
            log_probs_list.append(dist.log_prob(action_tensor))
        # sum log probs for each action dim
        total_log_prob = torch.stack(log_probs_list).sum(dim=0, keepdim=True)

        return actions, total_log_prob, hx_new, cx_new

    def get_value(self, obs_dict, hx=None, cx=None):
        projected_features = self._get_common_features(obs_dict)
        if hx is None: hx = torch.zeros(1, self.hidden_size, device=device)
        if cx is None: cx = torch.zeros(1, self.hidden_size, device=device)
        hx_new, _ = self.lstm(projected_features, (hx, cx))
        residual_input = hx_new
        x_after_residual = self.fc_residual(residual_input)
        x_after_residual = self.fc_residual_relu(residual_input + x_after_residual)
        value = self.value_head(x_after_residual)
        return value
    
    # used for re-evaluating states in PPO/ABC updates
    def forward_step(self, obs_dict, hx, cx):
        projected_features = self._get_common_features(obs_dict)
        hx_after_obs, _ = self.lstm(projected_features, (hx, cx))
        residual_input = hx_after_obs
        x_after_residual = self.fc_residual(residual_input)
        x_after_residual = self.fc_residual_relu(residual_input + x_after_residual)
        logits = self.policy_head(x_after_residual)
        value = self.value_head(x_after_residual)
        return logits, value

def evaluate_batch_recurrent(policy, obs_batch_list, hx_batch_tensor, cx_batch_tensor):
    all_logits, all_values = [], []
    for i in range(len(obs_batch_list)):
        obs_dict = obs_batch_list[i]
        hx_current = hx_batch_tensor[i].unsqueeze(0)
        cx_current = cx_batch_tensor[i].unsqueeze(0)
        logits, value = policy.forward_step(obs_dict, hx_current, cx_current)
        all_logits.append(logits) # logits is (1, sum_action_dims)
        all_values.append(value)  # value is (1, 1)
    batch_logits = torch.cat(all_logits, dim=0)     # (batch_size, sum_action_dims)
    batch_values = torch.cat(all_values, dim=0)     # (batch_size, 1)
    return batch_logits, batch_values

def ppo_loss(policy, obs_list, actions_tensor, old_log_probs_tensor, returns_tensor, advantages_tensor, hx_tensor, cx_tensor, clip_eps=0.2, ent_coef=0.01, vf_coef=1.0):
    # current policy's logits and values
    current_logits, current_values_raw = evaluate_batch_recurrent(policy, obs_list, hx_tensor, cx_tensor)
    current_values = current_values_raw.squeeze(-1)

    # split logits into separate categorical distributions
    split_logits = torch.split(current_logits, policy.action_dims, dim=-1)
    log_probs_list, entropy_list = [], []
    for i in range(len(policy.action_dims)):
        dist = Categorical(logits=split_logits[i])
        log_probs_list.append(dist.log_prob(actions_tensor[:, i]))
        entropy_list.append(dist.entropy())
    current_log_probs = torch.stack(log_probs_list, dim=-1).sum(dim=-1) # (batch_size,)
    entropy = torch.stack(entropy_list, dim=-1).sum(dim=-1)             # (batch_size,)

    # compute PPO loss
    ratio = torch.exp(current_log_probs - old_log_probs_tensor)
    surr1 = ratio * advantages_tensor
    surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps)*advantages_tensor
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = vf_coef*F.mse_loss(current_values, returns_tensor)
    entropy_bonus = ent_coef*entropy.mean()
    return policy_loss + value_loss - entropy_bonus

def abc_loss(policy, obs_list, actions_tensor, old_log_probs_tensor, hx_tensor, cx_tensor, clip_eps=0.2):
    # current policy's logits
    current_logits, _ = evaluate_batch_recurrent(policy, obs_list, hx_tensor, cx_tensor)

    # split logits and get current policy probs
    split_logits = torch.split(current_logits, policy.action_dims, dim=-1)
    log_probs_list = []
    for i in range(len(policy.action_dims)):
        dist = Categorical(logits=split_logits[i])
        log_probs_list.append(dist.log_prob(actions_tensor[:, i]))
    current_log_probs = torch.stack(log_probs_list, dim=-1).sum(dim=-1)

    # compute ABC loss
    ratio = torch.exp(current_log_probs - old_log_probs_tensor)
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    return -clipped_ratio.mean()

def ppo_train(policy, optimizer, batch):
    optimizer.zero_grad()
    loss = ppo_loss(policy, batch)
    loss.backward()
    optimizer.step()

def ppo_abc_train(policy, optimizer, batch_B, batch_BC, old_policy, beta=0.5):
    optimizer.zero_grad()
    loss = ppo_loss(policy, batch_B) + beta*abc_loss(policy, batch_BC)
    loss.backward()
    optimizer.step()

def compute_returns(rewards, gamma=0.998):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32, device=device)

def compute_gae(rewards, values, gamma=0.998, lam=0.95, last_val_bootstrap=0.0):
    advantages = []
    gae = 0.0
    extended_values = list(values) + [last_val_bootstrap]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma*extended_values[t+1] - extended_values[t]
        gae = delta + gamma*lam*gae
        advantages.insert(0, gae)
    returns = [adv + val for (adv, val) in zip(advantages, values)]
    return torch.tensor(advantages, dtype=torch.float32, device=device), torch.tensor(returns, dtype=torch.float32, device=device)
