
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import itertools

from zeroAlgo.randQLeraning import RandQLearning
from mcts_utils import MCTS

class MctsQLearning (RandQLearning):
    def choose_expl_action (self, state):
        """
        We use a Monte-Carlo Tree Search (MCTS) to compute a distribution over the action space.
        The value estimate for the leaves of the MCTS is provided by the value network.
        The prior for exploration over the actions at each node is uniform.
        """
        # print(state)
        mcts = MCTS(state, self.game_model, lambda x: self.calc_p(x), lambda next_state: self.value_network(self.game_model.generate_obs(next_state)).detach().numpy(), self.gamma)
        
        for i in range(30):
            # print(mcts.all_n, mcts.all_q, mcts.all_u)
            mcts.select()
            # print(mcts.new_q)
            # mcts.print_hierarchy ()
        action, action_dist = mcts.choose_action()
        return action, action_dist, mcts.all_q[action]

    def calc_p (self, state):
        return np.ones(self.game_model.n_action)/self.game_model.n_action

    def build_train_set (self, n_sample):
        """
        The training set is composed of triplets state-value-action_dist. 
        The states are sampled uniformly over the state space.
        Both the value target and the action distribution target are computed using the MCTS. 
        """
        all_obs = []
        all_values = []

        for sample in range(n_sample):
            state = self.game_model.reset()
            obs = self.game_model.generate_obs(state)
            action, action_dist, target_value = self.choose_expl_action(state)

            all_obs.append(obs)
            all_values.append(target_value)#rew + self.gamma * self.target_network(new_obs))
        
        all_obs = torch.cat(all_obs, dim=0)
        all_values = torch.tensor(np.stack(all_values))

        to_return = {"obs": all_obs, "values": all_values}
        return to_return