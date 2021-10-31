
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import itertools

from zeroAlgo.randQLeraning import RandQLearning

"""
Old relic while implementing the mcts algorithm.
"""

class onestepQLearning (RandQLearning):
    def choose_expl_action (self, state):
        expl_epsilon = 1#.25
        cpuct = 2

        next_states = [self.game_model.step(state, i) for i in range(self.game_model.n_action)]
        all_next_values = [self.value_network(self.game_model.generate_obs(next_state)).detach().numpy() for next_state in next_states]
        all_rew = [self.game_model.calc_rew(next_state) for next_state in next_states]
        

        all_p = np.array([(1-expl_epsilon) * 0 + expl_epsilon/self.game_model.n_action for action in range(self.game_model.n_action)])
        # all_q = np.zeros((self.game_model.n_action,))
        all_q = np.array(all_rew) + self.gamma * np.array(all_next_values).flatten()
        all_n = np.zeros((self.game_model.n_action,))
        all_w = np.zeros((self.game_model.n_action,))

        n_expl = 30
        for expl_step in range(n_expl):
            all_u = cpuct * all_p * np.sqrt(np.sum(all_n))/(1+all_n)
            a = np.argmax(all_q + all_u)
            next_state = self.game_model.step(state, a)
            next_value = self.game_model.calc_rew(next_state) + self.gamma * self.value_network(self.game_model.generate_obs(next_state)).detach().numpy()
            all_n[a] = all_n[a] + 1
            all_w[a] = all_w[a] + next_value
            all_q[a] = all_w[a] / all_n[a]

        # if np.all(state == np.array([0, 0, 0])):
        #     all_u = cpuct * all_p * np.sqrt(np.sum(all_n))/(1+all_n)
        #     print(all_n, all_q, all_u)


        tau = 1
        n_pow = np.power(all_n, 1/tau)
        choosen_action = np.random.choice(self.game_model.n_action, p=n_pow/np.sum(n_pow))
        return choosen_action
