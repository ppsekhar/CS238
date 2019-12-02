import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pdb

def clean_reward(reward_list): 
	return [int(re.sub("[^0-9]", "", elt))
			for elt in reward_list.strip().split(',')]

def format_grid(grid_str): 
	num_producers = []
	num_consumers = []
	energies = []
	budgets = []
	grid_squares = grid_str.split('),')
	for square in grid_squares:
		producers, consumers, energy, budget = square.split(',')
		producers, consumers, energy, budget = int(producers.replace('(','')), \
											   int(consumers), \
											   int(energy), \
											   int(budget.replace(')',''))
		num_producers.append(producers)
		num_consumers.append(consumers)
		energies.append(energy)
		budgets.append(budget)
	return [num_producers, num_consumers, energies, budgets]

def visualize_actions(grid_n_action, vis_energy=True, vis_budget=True): 

	#TODO: Programatically determine dimensions
	dim = 2
	if vis_energy: 
		eFig, eAxs = plt.subplots(dim) 
	if vis_budget: 
		bFig, bAxs = plt.subplots(dim)

	for i in range(len(grid_n_action)): 
		num_producers, num_consumers, energies, budgets = grid_n_action[i][0]
		action = grid_n_action[i][1]
		if vis_energy: 
			# TODO: Make sure access order is correct
			energies = np.array(energies)
			energies = energies.reshape((2,2))
			eAxs[i].imshow(energies, cmap='hot', interpolation='nearest')
			for i in range(2): 
				for j in range(2): 
					text = eAxs[i].text(i, j, str(energies[i][j]), \
						ha="center", va="center", color="w")
		if vis_budget:
			budgets = np.array(budgets)
			budgets = budgets.reshape((2,2))
			bAxs[i].imshow(budgets, cmap='hot', interpolation='nearest')
			for i in range(2): 
				for j in range(2): 
					text = bAxs[i].text(i, j, str(budgets[i][j]), \
						ha="center", va="center", color="w")

	plt.show()

def visualize_policy(policy_file, reward_thresh=0.5, num_vis=6): 
	extreme_reward = defaultdict(list)
	with open(policy_file, 'r') as fp: 
		line = fp.readline()
		while line:
			grid, action = line.split(":")
			reward = int(action.split(',')[2].strip()[:-1])
			if reward > reward_thresh or -reward > reward_thresh:
				extreme_reward[reward].append([format_grid(grid),clean_reward(action)])
   			line = fp.readline()

   	sorted_reward_keys = sorted(extreme_reward.keys())
   	actions_to_visualize = []

   	i = 0
   	while len(actions_to_visualize) < num_vis and len(extreme_reward) != 0 \
   			and i < len(sorted_reward_keys)//2:  

   		minimal_key = sorted_reward_keys[i]
   		maximal_key = sorted_reward_keys[len(sorted_reward_keys)-i-1]
   		if extreme_reward[minimal_key]: 
   			actions_to_visualize.append(extreme_reward[minimal_key].pop())
   			if extreme_reward[minimal_key] == 0: 
   				del extreme_reward[minimal_key]
   		if extreme_reward[maximal_key]: 
   			actions_to_visualize.append(extreme_reward[maximal_key].pop())
   			if extreme_reward[maximal_key] == 0: 
   				del extreme_reward[maximal_key]

   		i += 1
   	visualize_actions(actions_to_visualize)

visualize_policy('test_offline_out.policy',0.5)
