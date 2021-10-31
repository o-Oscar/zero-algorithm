
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import itertools

from zeroAlgo.randQLeraning import RandQLearning

class EpsilonQLearning (RandQLearning):
    def __init__ (self, game_model, expl_epsilon=0.1):
        super().__init__(game_model)
        self.expl_epsilon = expl_epsilon

    def choose_expl_action (self, state):
        """
        With probability self.expl_epsilon, the action is taken at random uniformly accross the action space.
        The rest of the time, we use the best available action. 
        """
        if np.random.random() < self.expl_epsilon:
            return np.random.randint(self.game_model.n_action)
        else:
            return self.choose_best_action(state)
