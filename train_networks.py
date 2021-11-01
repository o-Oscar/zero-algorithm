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




#Path(full_path).parent.mkdir(parents=True, exist_ok=True)

@dataclass
class Config:
	debug: bool
	result_folder: str
	model_name: str

	def get_path (self):
		debug_name = "debug_" if self.debug else ""
		full_path = os.path.join(debug_name+self.result_folder, self.model_name)
		return full_path
	
	def create_path (self, path_str):
		# Path(path_str).parent.mkdir(parents=True, exist_ok=True)
		Path(path_str).mkdir(parents=True, exist_ok=True)

	def get_all_dir (self):
		return sorted(next(os.walk(self.get_path()))[1])

def save_model (n_saves, config, qLearner):
	path_str = os.path.join(config.get_path(), "models_{}".format(n_saves))
	config.create_path(path_str)
	print("Saving model at : {}".format(path_str))

	qLearner.save(path_str)

def main ():

	debug = True

	game_model = HanoiTower (n_pallets=3)

	# qLearner = RandQLearning (game_model) ; config = Config(debug, "results", "randModel")
	# qLearner = EpsilonQLearning (game_model) ; config = Config(debug, "results", "epsilonModel")
	qLearner = mctsActionQLearning (game_model) ; config = Config(debug, "results", "mctsActionModel")

	start_time = time.time()

	epoch = 0

	max_training_time = 60
	min_save_time = 5
	last_save_time = -1
	n_saves = 0

	all_test_states = game_model.get_all_states() # [np.array([l, k, j, i]) for i, j, k, l in itertools.product(range(3), range(3), range(3), range(3))]
	all_test_obs = torch.cat([qLearner.game_model.generate_obs(state) for state in all_test_states], dim=0)
	to_plot = []

	while time.time() - start_time < max_training_time:
		to_plot.append(qLearner.value_network(all_test_obs).detach().numpy())
		
		advancement = (time.time() - start_time)/max_training_time
		advancement = 0 if advancement < 0 else (1 if advancement > 1 else advancement)
		qLearner.set_train_advancement(advancement)

		print("epoch {}".format(epoch))
		epoch += 1
		train_set = qLearner.build_train_set(100)
		qLearner.fit(train_set)

		if time.time() - last_save_time >= min_save_time:
			save_model (n_saves, config, qLearner)
			last_save_time = time.time()
			n_saves += 1
	

	to_plot = np.stack(to_plot)
	plt.plot(to_plot, "-")
	plt.show()
	


if __name__ == "__main__":
	main ()