from robogym.envs.rearrange import blocks_train
import torch
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import PPO
import copy
import pdb
from utils import *
from tqdm import tqdm
import random
import mujoco_py
import glfw
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

device = PPO.device
bob_successes = 0

def generate_alice_traj(env, policy, seed, T_alice, render=False):
    tau_A = []
    env.seed(seed)
    s0 = env.reset()
    initial_obj_pos = s0['obj_pos']
    s = s0
    num_objects = env.unwrapped.randomization.get_parameter("parameters:num_objects").get_value()
    clear_goal(env, num_objects)
    hx = torch.zeros(1, policy.hidden_size, device=device)
    cx = torch.zeros(1, policy.hidden_size, device=device)
    if render:
        viewer = mujoco_py.MjViewer(env.sim)
        viewer.cam.azimuth = 153.20427236315138
        viewer.cam.distance = 3.2007579769284864
        viewer.cam.elevation = -32.26301735647543
    for _ in range(T_alice):
        if render:
            viewer.render()
        s_hx = hx.detach().clone()
        s_cx = cx.detach().clone()
        action, log_prob, hx, cx = policy.get_action(s, hx, cx)
        log_prob = log_prob.detach()
        s_next, _, _, info = env.step(action)
        with torch.no_grad():
            value = policy.get_value(s, s_hx, s_cx).item()
        tau_A.append((s, action, 0, info, log_prob, value, s_hx, s_cx)) # no intermediate reward
        s = s_next
    
    final_obj_pos, final_obj_rot = s['obj_pos'], s['obj_rot']

    # Check whether any object has traveled for more than the success threshold
    valid_goal = np.any(np.linalg.norm(final_obj_pos - initial_obj_pos, axis=1) > 0.04)

    # Check whether all the objects are on the table
    if valid_goal:
        valid_goal = np.all(info['objects_off_table'] == False)

    if valid_goal:
        distance = np.linalg.norm(final_obj_pos - initial_obj_pos)
        #tau_A[-1][0]['is_goal_achieved'][:] = 1
        print(f'Object moved by {distance:.4f} m')

    if render and viewer.window:
        glfw.destroy_window(viewer.window)

    goal_in_placement_area = env.unwrapped.mujoco_simulation.check_objects_in_placement_area(final_obj_pos)

    return tau_A, final_obj_pos, final_obj_rot, valid_goal, goal_in_placement_area

def generate_bob_traj(env, policy, goal_obj_pos, goal_obj_rot, max_steps_bob, seed, render=False):
    tau_B = []
    env.seed(seed)
    s0 = env.reset()
    s = s0.copy()

    # Set goal
    s['goal_obj_pos'] = goal_obj_pos
    #s['goal_obj_rot'] = goal_obj_rot
    s['rel_goal_obj_pos'] = goal_obj_pos - s['obj_pos']
    #s['rel_goal_obj_rot'] = goal_obj_rot - s['obj_rot']
    
    xi = False
    num_objects = env.unwrapped.randomization.get_parameter("parameters:num_objects").get_value()
    object_at_goal = np.zeros(num_objects, dtype=bool)
    sync_goal(env, goal_obj_pos, num_objects)
    hx = torch.zeros(1, policy.hidden_size, device=device)
    cx = torch.zeros(1, policy.hidden_size, device=device)
    if render:
        viewer = mujoco_py.MjViewer(env.sim)
        viewer.cam.azimuth = 153.20427236315138
        viewer.cam.distance = 3.2007579769284864
        viewer.cam.elevation = -32.26301735647543
    for _ in range(max_steps_bob):
        if render:
            viewer.render()
        s_hx = hx.detach().clone()
        s_cx = cx.detach().clone()
        action, log_prob, hx, cx = policy.get_action(s, hx, cx)
        log_prob = log_prob.detach()
        s_next_raw, _, done, info = env.step(action)
        s_next = s_next_raw.copy()
        
        # Maintain goal state
        s_next['goal_obj_pos'] = goal_obj_pos
        #s_next['goal_obj_rot'] = goal_obj_rot
        s_next['rel_goal_obj_pos'] = goal_obj_pos - s_next['obj_pos']
        #s_next['rel_goal_obj_rot'] = goal_obj_rot - s_next['obj_rot']
        
        #tau_B.append((s, action, 0, info, log_prob))      # Overwrite original reward later
        #s = s_next

        s_reward = 0
        all_objects_at_goal = True
        current_obj_pos = s_next['obj_pos']
        current_obj_rot = s_next['obj_rot']
        for i in range(num_objects):
            # pos check
            pos_error = np.linalg.norm(current_obj_pos[i] - goal_obj_pos[i])
            is_pos_correct = pos_error < 0.04

            # rot check
            r_current = Rotation.from_euler('xyz', current_obj_rot[i])
            r_goal = Rotation.from_euler('xyz', goal_obj_rot[i])
            relative_rot = r_goal * r_current.inv()
            rot_error = np.linalg.norm(relative_rot.as_rotvec())
            #rot_error = goal_obj_rot[i] - current_obj_rot[i]
            is_rot_correct = rot_error < 0.2

            #object_at_goal_now = is_pos_correct and is_rot_correct
            object_at_goal_now = is_pos_correct
            if object_at_goal_now.all():
                if not object_at_goal[i]:
                    s_reward += 1   # arrived for the first time or after moving away
                    object_at_goal[i] = True
            else:
                if object_at_goal[i]:
                    s_reward -= 1   # moved away after having arrived
                    object_at_goal[i] = False
                all_objects_at_goal = False

        if all_objects_at_goal:
            s_reward += 5
            xi = True

        with torch.no_grad():
            value = policy.get_value(s, s_hx, s_cx).item()

        tau_B.append((s, action, s_reward, info, log_prob, value, s_hx, s_cx))
        s = s_next

        if xi or done:
            break

    if render and viewer.window:
        glfw.destroy_window(viewer.window)

    return tau_B, xi

def compute_alice_reward(xi, valid_goal, goal_in_placement_area):
    r_A = 0
    if valid_goal:
        r_A = 1

        # Reward Alice if Bob failed
        if not xi:
            r_A += 5

        # Penalize Alice if it pushed any object out of the placement area
        if np.any(goal_in_placement_area == False):
            r_A -= 3
            
    return r_A

def relabel_demonstration(tau_A, goal_obj_pos, goal_obj_rot, old_bob_policy):
    tau_BC = []
    current_hx_bob = torch.zeros(1, old_bob_policy.hidden_size, device=device)
    current_cx_bob = torch.zeros(1, old_bob_policy.hidden_size, device=device)
    for (s_alice, a_alice, _, _, _, _, _, _) in tau_A:
        s_augmented = s_alice.copy()
        s_augmented['goal_obj_pos'] = goal_obj_pos
        #s_augmented['goal_obj_rot'] = goal_obj_rot
        s_augmented['rel_goal_obj_pos'] = goal_obj_pos - s_alice['obj_pos']
        #s_augmented['rel_goal_obj_rot'] = goal_obj_rot - s_alice['obj_rot']
        log_prob_alice_under_bob_old = torch.tensor(0.0, device=device)
        with torch.no_grad():
            projected_features_bob = old_bob_policy._get_common_features(s_augmented)
            hx_bob_after_obs, cx_bob_after_obs = old_bob_policy.lstm(projected_features_bob, (current_hx_bob, current_cx_bob))
            residual_input_bob = hx_bob_after_obs
            x_after_residual_bob = old_bob_policy.fc_residual(residual_input_bob)
            x_after_residual_bob = old_bob_policy.fc_residual_relu(residual_input_bob + x_after_residual_bob)
            logits_bob_old = old_bob_policy.policy_head(x_after_residual_bob)
            split_logits_bob_old = torch.split(logits_bob_old.squeeze(0), old_bob_policy.action_dims, dim=-1)
            log_probs_list_bob_old = []
            for i_dim, logits_for_dim in enumerate(split_logits_bob_old):
                action_component_tensor = torch.tensor(a_alice[i_dim], device=device, dtype=torch.long)
                dist = Categorical(logits=logits_for_dim)
                log_probs_list_bob_old.append(dist.log_prob(action_component_tensor))
            log_prob_alice_under_bob_old = torch.stack(log_probs_list_bob_old).sum(dim=0, keepdim=True)
        tau_BC.append((s_augmented, a_alice, 0, log_prob_alice_under_bob_old.detach(), current_hx_bob.detach(), current_cx_bob.detach()))
        current_hx_bob = hx_bob_after_obs.detach()
        current_cx_bob = cx_bob_after_obs.detach()
    return tau_BC

# Convert demonstrations into tensors for ABC training
def prepare_demo(demo_rollouts):
    augmented_states, actions, log_probs, hx_demos, cx_demos = [], [], [], [], []
    for traj_bc in demo_rollouts:
        for (s_aug, a_alice, _, lp_alice, hx_s, cx_s) in traj_bc:
            augmented_states.append(s_aug)
            actions.append(torch.tensor(a_alice, dtype=torch.long, device=device))
            log_probs.append(lp_alice)
            hx_demos.append(hx_s)
            cx_demos.append(cx_s)

    if not augmented_states:
        return None
    
    actions_tensor = torch.stack(actions)
    lp_tensor = torch.stack(log_probs)
    if lp_tensor.ndim > 1 and lp_tensor.shape[-1] == 1:
        lp_tensor = lp_tensor.squeeze(-1)
    hx_demos_tensor = torch.cat(hx_demos, dim=0)
    cx_demos_tensor = torch.cat(cx_demos, dim=0)
    return augmented_states, actions_tensor, lp_tensor, hx_demos_tensor, cx_demos_tensor

# Algo 2: CollectRolloutData
def collect_rollout_data(alice_env, bob_env, alice, bob, render=False):
    global bob_successes
    # Initialize empty replay buffer
    D_A, D_B, D_BC = [], [], []
    # Initialize Bob to success
    bob_can_continue_this_episode = True

    T_alice = 100
    max_steps_bob = 200

    bob_successes_this_call = 0
    bob_attempts_this_call = 0

    # 5-goal Asymmetric Self-Play episode
    for round in range(5):
        seed = np.random.randint(low=0, high=2**32)

        # Alice's turn
        tau_A, g_pos, g_rot, valid_goal, goal_in_placement_area = generate_alice_traj(alice_env, alice, seed=seed, T_alice=T_alice, render=render)
        if not valid_goal:
            #print(f'Goal #{round+1} not valid')
            break
        print(f'Goal #{round+1} valid')

        # Bob's turn (executed only if Bob has not failed)
        xi_this_turn = False
        bob_played_this_turn = False
        if bob_can_continue_this_episode:
            bob_played_this_turn = True
            bob_attempts_this_call += 1
            # Generate Bob's trajectory
            tau_B, xi = generate_bob_traj(bob_env, bob, g_pos, g_rot, max_steps_bob=max_steps_bob, seed=seed, render=render)
            # Update Bob's replay buffer
            if tau_B:
                D_B.append(tau_B)
            if xi:
                xi_this_turn = True
                bob_successes_this_call += 1
            else:
                bob_can_continue_this_episode = False

        # Compute Alice's reward
        r_A = compute_alice_reward(xi_this_turn, valid_goal, goal_in_placement_area)
        # Overwrite the last reward in tau_A with r_A
        s_last, a_last, _, info_last, lp_last, val_last, hx_last, cx_last = tau_A[-1]
        tau_A[-1] = (s_last, a_last, r_A, info_last, lp_last, val_last, hx_last, cx_last)
        #tau_A[-1] = (tau_A[-1][0], tau_A[-1][1], r_A, tau_A[-1][3], tau_A[-1][4])

        if tau_B:
            print(f'Alice reward: {tau_A[-1][2]} | Bob reward: {tau_B[-1][2]}')
        else:
            print(f'Alice reward: {tau_A[-1][2]} | Bob skipped turn')

        # Update Alice's replay buffer
        D_A.append(tau_A)

        # If Bob failed, update ABC replay buffer
        if bob_played_this_turn and not xi_this_turn:
            tau_BC = relabel_demonstration(tau_A, g_pos, g_rot, bob)
            D_BC.append(tau_BC)

    return D_A, D_B, D_BC, bob_successes_this_call, bob_attempts_this_call

# Convert trajectory rollouts into tensors for PPO training
def prepare_batch(rollouts, gamma=0.998, lam=0.95):
    batch_obs_dicts, batch_actions, batch_old_log_probs, batch_returns_list, batch_advantages_list = [], [], [], [], []
    batch_initial_hx, batch_initial_cx = [], []
    
    for traj in rollouts:
        traj_rewards, traj_values = [], [] # for GAE
        for (s, a, r, _info, log_prob_old, value_old, hx_s, cx_s) in traj:
            batch_obs_dicts.append(s)
            batch_actions.append(torch.as_tensor(a, dtype=torch.long, device=device))
            batch_old_log_probs.append(log_prob_old)
            traj_rewards.append(r)
            traj_values.append(value_old)
            batch_initial_hx.append(hx_s)
            batch_initial_cx.append(cx_s)

        last_val_bootstrap = 0.0
        advantages_traj, returns_traj = PPO.compute_gae(traj_rewards, traj_values, gamma=gamma, lam=lam, last_val_bootstrap=last_val_bootstrap)
        batch_returns_list.append(returns_traj)
        batch_advantages_list.append(advantages_traj)

    if not batch_obs_dicts:
        return None
    
    batch_actions_tensor = torch.stack(batch_actions)
    batch_old_log_probs_tensor = torch.stack(batch_old_log_probs)
    batch_returns_tensor = torch.cat(batch_returns_list, dim=0).to(dtype=torch.float32, device=device)
    batch_advantages_tensor = torch.cat(batch_advantages_list, dim=0).to(dtype=torch.float32, device=device)
    batch_initial_hx_tensor = torch.cat(batch_initial_hx, dim=0)
    batch_initial_cx_tensor = torch.cat(batch_initial_cx, dim=0)

    return batch_obs_dicts, batch_actions_tensor, batch_old_log_probs_tensor, batch_returns_tensor, batch_advantages_tensor, batch_initial_hx_tensor, batch_initial_cx_tensor

# Algo 1: Asymmetric Self-Play
def asp(alice, bob, n_updates=3, training_steps=10, batch_size=4096, render=False):
    global bob_successes
    alice_optimizer = optim.Adam(alice.parameters(), lr=3e-4)
    bob_optimizer = optim.Adam(bob.parameters(), lr=3e-4)
    
    history = {
        'bob_success_rates': [],
        'alice_ppo_loss': [],
        'bob_ppo_loss': [],
        'bob_abc_loss': [],
        'alice_env_reward_mean': [],
        'bob_env_reward_sum_mean': [],
        'alice_gae_return_mean': [],
        'bob_gae_return_mean': []
    }

    # History pools for snapshots of past policies
    past_alice, past_bob = [], [] 
    for step in range(training_steps):
        print(f'\n=== Training step {step+1}/{training_steps} ===')

        # Initialize environments
        alice_env, bob_env = init_envs()
        #num_objects = alice_env.unwrapped.randomization.get_parameter("parameters:num_objects").get_value()
        #print(f'[asp] Number of objects: {num_objects}')

        selected_alice = select_policy(alice, past_alice, prob=0.2)
        selected_bob = select_policy(bob, past_bob, prob=0.2)

        # Collect batch data
        batch_DA, batch_DB, batch_DBC = [], [], []
        total_transitions = 0
        current_step_total_bob_successes = 0
        current_step_total_bob_attempts = 0
        pbar = tqdm(total=batch_size, desc='Collecting transitions', unit='transitions')
        while total_transitions < batch_size:
            D_A, D_B, D_BC, successes_in_call, attempts_in_call = collect_rollout_data(alice_env, bob_env, selected_alice, selected_bob, render=render)

            batch_DA.extend(D_A)
            batch_DB.extend(D_B)
            batch_DBC.extend(D_BC)

            current_step_total_bob_successes += successes_in_call
            current_step_total_bob_attempts += attempts_in_call

            new_transitions = (
                sum(len(episode) for episode in D_A) +
                sum(len(episode) for episode in D_B if episode) +
                sum(len(episode) for episode in D_BC if episode)
            )
            total_transitions += new_transitions
            pbar.update(new_transitions)
        pbar.close()

        current_step_bob_success_rate = (100*current_step_total_bob_successes/current_step_total_bob_attempts) if current_step_total_bob_attempts > 0 else 0
        history["bob_success_rates"].append(current_step_bob_success_rate)
        print(f'Bob success rate for this batch: {current_step_bob_success_rate:.2f}% ({current_step_total_bob_successes}/{current_step_total_bob_attempts})')
        # Convert rollouts into batches
        batch_A = prepare_batch(batch_DA) if batch_DA else None
        batch_B = prepare_batch(batch_DB) if batch_DB else None
        batch_BC_data = prepare_demo(batch_DBC) if batch_DBC else None

        if batch_A:
            history["alice_gae_return_mean"].append(batch_A[3].mean().item())
            alice_env_rewards_this_step = [traj[-1][2] for traj in batch_DA if traj]
            history["alice_env_reward_mean"].append(np.mean(alice_env_rewards_this_step) if alice_env_rewards_this_step else 0)
        else:
            history["alice_gae_return_mean"].append(0)
            history["alice_env_reward_mean"].append(0)

        if batch_B:
            history["bob_gae_return_mean"].append(batch_B[3].mean().item())
            bob_env_rewards_sums_this_step = [sum(step_data[2] for step_data in traj) for traj in batch_DB if traj]
            history["bob_env_reward_sum_mean"].append(np.mean(bob_env_rewards_sums_this_step) if bob_env_rewards_sums_this_step else 0)
        else:
            history["bob_gae_return_mean"].append(0)
            history["bob_env_reward_sum_mean"].append(0)

        current_step_alice_ppo_losses = []
        current_step_bob_ppo_losses = []
        current_step_bob_abc_losses = []
        for update in range(n_updates): # Sample reuse
            print(f'Update {update+1}/{n_updates}')
            
            # Update Alice
            if batch_A:
                alice_loss = PPO.ppo_loss(alice, batch_A[0], batch_A[1], batch_A[2], batch_A[3], batch_A[4], batch_A[5], batch_A[6])
                alice_optimizer.zero_grad()
                alice_loss.backward()
                alice_optimizer.step()
                current_step_alice_ppo_losses.append(alice_loss.item())

            # Update Bob
            if batch_B:
                bob_ppo_loss = PPO.ppo_loss(bob, batch_B[0], batch_B[1], batch_B[2], batch_B[3], batch_B[4], batch_B[5], batch_B[6])
                current_step_bob_ppo_losses.append(bob_ppo_loss.item())
                bob_loss = bob_ppo_loss
                if batch_BC_data:
                    bob_abc_loss = PPO.abc_loss(bob, batch_BC_data[0], batch_BC_data[1], batch_BC_data[2], batch_BC_data[3], batch_BC_data[4])
                    current_step_bob_abc_losses.append(bob_abc_loss.item())
                    bob_loss += 0.5*bob_abc_loss
                bob_optimizer.zero_grad()
                bob_loss.backward()
                bob_optimizer.step()

        history["alice_ppo_loss"].append(np.mean(current_step_alice_ppo_losses) if current_step_alice_ppo_losses else 0)
        history["bob_ppo_loss"].append(np.mean(current_step_bob_ppo_losses) if current_step_bob_ppo_losses else 0)
        history["bob_abc_loss"].append(np.mean(current_step_bob_abc_losses) if current_step_bob_abc_losses else 0)

        print_diagnostics(history, step)

        torch.save(alice.state_dict(), 'alice.pth')
        torch.save(bob.state_dict(), 'bob.pth')

        # Save current snapshots of policies
        past_alice.append(copy.deepcopy(alice).cpu())
        past_bob.append(copy.deepcopy(bob).cpu())
        if len(past_alice) > 10: past_alice.pop(0)
        if len(past_bob) > 10: past_bob.pop(0)
        alice_env.close()
        bob_env.close()

        if (step + 1) % 10 == 0 or step == training_steps - 1:
            plot_training_diagnostics(history, training_steps, save_path_prefix=f'training_diagnostics_step_{step+1}')

    return history

def print_diagnostics(history, step):
    print(f"--- Step {step+1} Diagnostics ---")
    print(f"  Bob Success Rate: {history['bob_success_rates'][-1]:.2f}%")
    if history['alice_ppo_loss']: print(f"  Alice PPO Loss: {history['alice_ppo_loss'][-1]:.4f}")
    if history['bob_ppo_loss']: print(f"  Bob PPO Loss: {history['bob_ppo_loss'][-1]:.4f}")
    if history['bob_abc_loss'][-1] != 0 : print(f"  Bob ABC Loss: {history['bob_abc_loss'][-1]:.4f}") # Only print if ABC was active
    if history['alice_env_reward_mean']: print(f"  Alice Env Reward Mean: {history['alice_env_reward_mean'][-1]:.3f}")
    if history['bob_env_reward_sum_mean']: print(f"  Bob Env Reward Sum Mean: {history['bob_env_reward_sum_mean'][-1]:.3f}")
    if history['alice_gae_return_mean']: print(f"  Alice GAE Return Mean: {history['alice_gae_return_mean'][-1]:.3f}")
    if history['bob_gae_return_mean']: print(f"  Bob GAE Return Mean: {history['bob_gae_return_mean'][-1]:.3f}")
    print("-------------------------")

def plot_training_diagnostics(history, total_training_steps, save_path_prefix='training_diagnostics'):
    num_metrics = len([k for k, v in history.items() if v])
    if num_metrics == 0:
        print("No data in history to plot.")
        return

    cols = 2
    rows = (num_metrics + cols - 1) // cols 

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 7, rows * 5), squeeze=False)
    axs = axs.flatten()

    plot_idx = 0
    training_steps_x = np.arange(1, len(history.get("bob_success_rates", [])) + 1) # Common x-axis

    # Plot Bob's Success Rate
    if history.get("bob_success_rates"):
        ax = axs[plot_idx]
        ax.plot(training_steps_x, history["bob_success_rates"], 'o-', label='Bob Success Rate (%)', alpha=0.7)
        # Rolling mean for success rate
        if len(history["bob_success_rates"]) >= 10:
            rolling_mean_success = np.convolve(history["bob_success_rates"], np.ones(10)/10, mode='valid')
            ax.plot(training_steps_x[9:], rolling_mean_success, 'r-', label='10-step Rolling Mean Success')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Rate (%)')
        ax.set_title('Bob Success Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Plot Losses
    loss_keys = ["alice_ppo_loss", "bob_ppo_loss", "bob_abc_loss"]
    loss_labels = ["Alice PPO Loss", "Bob PPO Loss", "Bob ABC Loss"]
    for key, label in zip(loss_keys, loss_labels):
        if history.get(key) and any(history[key]):
            ax = axs[plot_idx]
            ax.plot(training_steps_x[:len(history[key])], history[key], 'o-', label=label, alpha=0.7, markersize=3)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Loss')
            ax.set_title(label)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1
            
    # Plot Environment Rewards/Returns
    env_reward_keys = ["alice_env_reward_mean", "bob_env_reward_sum_mean"]
    env_reward_labels = ["Alice Avg Env Reward (per goal)", "Bob Avg Env Return (per attempt)"]
    for key, label in zip(env_reward_keys, env_reward_labels):
        if history.get(key):
            ax = axs[plot_idx]
            ax.plot(training_steps_x, history[key], 'o-', label=label, alpha=0.7)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Reward/Return')
            ax.set_title(label)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1
            
    # Plot GAE Returns
    gae_return_keys = ["alice_gae_return_mean", "bob_gae_return_mean"]
    gae_return_labels = ["Alice Avg GAE Return", "Bob Avg GAE Return"]
    for key, label in zip(gae_return_keys, gae_return_labels):
        if history.get(key):
            ax = axs[plot_idx]
            ax.plot(training_steps_x, history[key], 'o-', label=label, alpha=0.7)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('GAE Return')
            ax.set_title(label)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1

    # Hide any unused subplots
    for i in range(plot_idx, len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_diagnostics.png")
    print(f"Saved diagnostic plots to {save_path_prefix}_diagnostics.png")
    plt.close(fig)    

def plot_success_rates(success_rates, save_path='bob_success_rates.png'):
    """
    Plot Bob's success rates with a rolling mean for better readability
    
    Args:
        success_rates: List of success rates
        save_path: File path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Plot raw data points
    episodes = np.arange(1, len(success_rates) + 1)
    plt.plot(episodes, success_rates, 'o', alpha=0.4, color='lightblue', label='Raw success rate')
    
    # Calculate and plot rolling mean (if we have enough data)
    if len(success_rates) >= 10:
        rolling_mean = []
        for i in range(len(success_rates)):
            start_idx = max(0, i - 9)  # Take up to 10 previous episodes
            window = success_rates[start_idx:i+1]
            rolling_mean.append(np.mean(window))
        
        plt.plot(episodes, rolling_mean, 'b-', linewidth=2, label='10-episode rolling mean')
    
    plt.title('Bob\'s Success Rate Over Training', fontsize=14)
    plt.xlabel('Training Episode', fontsize=12)
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.ylim(0, 105)  # Leave a little space above 100%
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    
    # Add tick marks every 10 episodes for better readability
    max_episode = len(success_rates)
    tick_spacing = 10 if max_episode > 20 else 5
    plt.xticks(np.arange(0, max_episode + tick_spacing, tick_spacing))
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    

def select_policy(current_policy, past_policies, prob=0.2):
    if past_policies and np.random.rand() < prob:
        selected = random.choice(past_policies)
        print('Using past opponents')
        return selected
    return current_policy
    
def load_models(action_dims):
    alice = PPO.Policy(action_dims)
    bob = PPO.Policy(action_dims, is_goal_conditioned=True)
    alice.load_state_dict(torch.load('alice.pth'))
    bob.load_state_dict(torch.load('bob.pth'))
    alice.to(device)
    bob.to(device)
    return alice, bob

def test_policy(policy, env, n_tests=5, render=True):
    for _ in range(n_tests):
        s = env.reset()
        done = False
        hx, cx = None, None
        if render:
            viewer = mujoco_py.MjViewer(env.sim)
            viewer.cam.azimuth = 153.20427236315138
            viewer.cam.distance = 3.2007579769284864
            viewer.cam.elevation = -32.26301735647543
        while not done:
            if render:
                viewer.render()
            action, _, hx, cx = policy.get_action(s, hx, cx)
            s_next, reward, done, info = env.step(action)
                        
            s = s_next

def init_envs():
    # Create environments
    alice_env = blocks_train.make_env(parameters={'simulation_params': {'max_num_objects': 2}})
    bob_env = blocks_train.make_env(parameters={'simulation_params': {'max_num_objects': 2}})

    # Randomize number of objects
    #num_objects = np.random.randint(1, 3)
    num_objects = 1
    alice_env.unwrapped.randomization.get_parameter("parameters:num_objects").set_value(num_objects)
    bob_env.unwrapped.randomization.get_parameter("parameters:num_objects").set_value(num_objects)

    # Determine observation dimensions from a sample observation
    #sample_obs = alice_env.reset()
    #obs_tensor = PPO.process_obs(sample_obs)
    #obs_dim = obs_tensor.shape[0]
    #action_dims = [11, 11, 11, 11, 11, 11]

    # Determine goal dimension from a sample goal
    #goal_sample = sample_obs['goal_obj_pos']
    #goal_dim = np.array(goal_sample).flatten().shape[0]
    
    #return alice_env, bob_env, obs_dim, goal_dim, action_dims
    return alice_env, bob_env

def test_asymmetric_self_play(alice, bob, num_episodes=5, render=True):
    alice.eval()
    bob.eval()
    
    success_stats = []
    
    for episode in range(num_episodes):
        print(f"\n=== Testing Episode {episode+1}/{num_episodes} ===")
        
        # Create fresh environments for Alice and Bob
        alice_env, bob_env = init_envs()
        
        # Track Bob's performance for this episode
        episode_goals = 0
        episode_successes = 0
        bob_xi = True  # Bob starts as successful
        
        # Play 5 goals per episode
        for goal_num in range(5):
            seed = np.random.randint(low=0, high=2**32)
            
            # Let Alice generate a goal
            tau_A, goal, final_obj_rot, valid_goal, goal_in_placement_area = generate_alice_traj(
                alice_env, alice, seed=seed, render=render)
            
            if not valid_goal:
                print(f"Goal #{goal_num+1}: Invalid - Alice failed to generate a valid goal")
                break
                
            print(f"Goal #{goal_num+1}: Valid - Alice moved objects by {np.linalg.norm(goal - tau_A[0][0]['obj_pos']):.4f} m")
            if np.any(goal_in_placement_area == False):
                print(f"PENALTY: Alice pushed objects out of the placement area")
            episode_goals += 1
            
            # Bob's turn (only if Bob hasn't failed yet)
            if bob_xi:
                # Let Bob try to solve the goal
                tau_B, bob_xi = generate_bob_traj(
                    bob_env, bob, goal, final_obj_rot, seed=seed, render=render)
                
                if bob_xi:
                    print(f"Goal #{goal_num+1}: Bob succeeded in {len(tau_B)} steps")
                    episode_successes += 1
                else:
                    print(f"Goal #{goal_num+1}: Bob failed after {len(tau_B)} steps")
            else:
                print(f"Goal #{goal_num+1}: Bob skipped (already failed)")
        
        # Calculate success rate for this episode
        if episode_goals > 0:
            episode_success_rate = 100 * episode_successes / episode_goals
            success_stats.append((episode_goals, episode_successes, episode_success_rate))
            print(f"Episode Summary: {episode_successes}/{episode_goals} goals achieved ({episode_success_rate:.1f}%)")
        else:
            print("Episode Summary: No valid goals generated")
            
        # Close environments
        alice_env.close()
        bob_env.close()
    
    # Print overall statistics
    total_goals = sum(stat[0] for stat in success_stats)
    total_successes = sum(stat[1] for stat in success_stats)
    if total_goals > 0:
        overall_success_rate = 100 * total_successes / total_goals
        print(f"\nOverall Performance: {total_successes}/{total_goals} goals achieved ({overall_success_rate:.1f}%)")
    else:
        print("\nOverall Performance: No valid goals generated")
