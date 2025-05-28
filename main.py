from robogym.envs.rearrange import blocks, ycb, composer, mixture, blocks_train
from asp import asp, init_envs, load_models, test_policy, test_asymmetric_self_play
import PPO
import numpy as np

print(PPO.device)

action_dims = [11, 11, 11, 11, 11, 11]
#num_objects = alice_env.unwrapped.randomization.get_parameter("parameters:num_objects").get_value()
#print(f'[main] Number of objects: {num_objects}')
alice = PPO.Policy(action_dims=action_dims).to(PPO.device)
bob = PPO.Policy(action_dims=action_dims, is_goal_conditioned=True).to(PPO.device)

train = True
load = True
render = False
test_asp = False
test_single_policy = False

if train:
    training_steps = 1000
    if load:
        alice, bob = load_models(action_dims)
    asp(alice, bob, n_updates=3, training_steps=training_steps, batch_size=4096, render=render)
    print('Training completed.')
elif test_asp:
    print("Testing Alice and Bob in asymmetric self-play")
    alice, bob = load_models(action_dims)
    alice.eval()
    bob.eval()
    test_asymmetric_self_play(alice, bob, num_episodes=5, render=True)
else:
    n_tests = 5
    alice, bob = load_models(action_dims)
    alice.eval()
    bob.eval()
    test_env = blocks.make_env(parameters={'simulation_params': {'num_objects': 2, 'max_num_objects': 8}})
    test_policy(bob, test_env, n_tests=n_tests, render=True)
