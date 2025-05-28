import pdb

def print_dict(dictionary):
	for key, value in dictionary.items():
		print(f'{key}: {value}')

def set_state(old_state: dict, new_state: dict):
	for key, value in new_state.items():
		old_state[key] = value
	return old_state

def sync_goal(env, goal, num_objects):
	goal = goal[::-1]
	for i in range(num_objects):
		env.sim.model.body_pos[-(2*i + 1)] = goal[i]

def clear_goal(env, num_objects):
	for i in range(1, num_objects*2, 2):
		env.sim.model.body_pos[-i][:] = [-10, -10, -10]