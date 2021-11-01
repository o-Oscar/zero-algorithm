import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import itertools
import time
from dataclasses import dataclass
from pathlib import Path
import os

from zeroAlgo.randQLeraning import RandQLearning
from zeroAlgo.epsilonQLearning import EpsilonQLearning
from zeroAlgo.mctsQLearning import MctsQLearning
from zeroAlgo.mctsActionQLearning import mctsActionQLearning

from game.hanoi import HanoiTower

from train_networks import Config


# def eval_learner (qLearner, config, ):


def main ():

	debug = True

	game_model = HanoiTower (n_pallets=3)

	qLearner = mctsActionQLearning (game_model) ; config = Config(debug, "results", "mctsActionModel")

	all_test_states = game_model.get_interesting_states()
	best_next_states = np.array([all_test_states[0]] + all_test_states[:-1])
	for model_name in config.get_all_dir():
		qLearner.load(os.path.join(config.get_path(), model_name))

		n_good = 0
		for state, target_next_state in zip(all_test_states, best_next_states):
			print(state, end="")#next_state)
			for i in range(1):
				action = qLearner.choose_best_action(state)
				state = game_model.step(state, action)
				print("->", state, end="")#next_state)
			print()

			if np.all(state == target_next_state):
				n_good += 1
		accuracy = n_good / len(all_test_states)
		print(accuracy)
				
	
	


if __name__ == "__main__":
	main ()