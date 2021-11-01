
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import itertools
import os

from networks import ValueNetwork

class RandQLearning:
    def __init__ (self, game_model):
        self.game_model = game_model

        self.value_network = ValueNetwork(self.game_model.obs_dims)
        self.target_value_network = ValueNetwork(self.game_model.obs_dims)
        self.target_value_network.load_state_dict(self.value_network.state_dict())

        self.optimizer = optim.Adam(self.value_network.parameters(), lr=0.001)

        self.gamma = self.game_model.gamma # 0.97 # the gamma is part of the thing that we wish to maximize. It is part of the environment

        self.value_epsilon = 0.01

    def choose_expl_action (self, state):
        """
        Simply chooses a random action as a mean to explore and exloit. (Thus, in this case, there is no exploitation.)
        """
        return np.random.randint(self.game_model.n_action)

    def choose_best_action (self, state):
        """
        Uses a one-deep tree to evaluate the best action to take. The root is the current state. The leaves are the next states.
        This algorithm uses its internal trained value-function to evaluate the next states values and incidentally, each action's value. 
        """
        next_states = [self.game_model.step(state, i) for i in range(self.game_model.n_action)]
        all_next_values = [self.value_network(self.game_model.generate_obs(next_state)).detach().numpy() for next_state in next_states]
        all_rew = [self.game_model.calc_rew(next_state) for next_state in next_states]
        action = np.argmax([rew + self.gamma * value for rew, value in zip(all_rew, all_next_values)])
        return action

    def calc_value (self, state):
        obs = self.game_model.generate_obs (state)
        return self.value_network(obs)
    
    def build_train_set (self, n_sample):
        """
        Generates an ensemble of state - values pairs to train the value network.
        The states are sampled uniformly over the state space.
        It uses its best estimate of the value of the state by choosing the best action and then looking at the predicted value after taking this action.
        """
        all_obs = []
        all_values = []
        for sample in range(n_sample):
            state = self.game_model.reset()
            obs = self.game_model.generate_obs(state)
            action = self.choose_expl_action(state)
            new_state = self.game_model.step(state, action)
            new_obs = self.game_model.generate_obs(new_state)
            rew = self.game_model.calc_rew(new_state)

            all_obs.append(obs)
            all_values.append(rew + self.gamma * self.target_value_network(new_obs))
        
        all_obs = torch.cat(all_obs, dim=0)
        all_values = torch.cat(all_values, dim=0).detach()

        to_return = {"obs": all_obs, "values": all_values}
        return to_return
    
    def fit (self, train_set):
        """
        Fits the value network to the dataset previously generated.
        """
        all_obs = train_set["obs"]
        all_values = train_set["values"]
        n_sub_epoch = 100
        all_loss = []

        init_value = self.value_network(all_obs).detach()

        for epoch in range(n_sub_epoch):
            self.optimizer.zero_grad()
            
            cur_value = self.value_network(all_obs)
            loss = torch.mean(torch.square(cur_value - all_values) + torch.square(cur_value-init_value) / self.value_epsilon)
            loss.backward()
            all_loss.append(loss.detach().numpy())
            self.optimizer.step()
        
        self.fit_traget_network ()
    
    def fit_traget_network (self, retention=0.2):
        """
        This version of the algorithm uses a target value network to stabilize the q-learning procedure.
        We slowly update the weights of the target value network toward the weights of the trained value network.
        This target value network changes slowly and thus offers more stable values to be used to bootstrap the training of the actual value network.
        A low retention (\in [0,1]) changes the target value network each iteration a lot. 
        """
        new_target_dict = {}
        for (key, cur_value), (_, target_value) in zip(self.value_network.state_dict().items(), self.target_value_network.state_dict().items()):
            new_target_dict[key] = retention * target_value + (1-retention)* cur_value
        self.target_value_network.load_state_dict(new_target_dict)

    def save (self, base_path):
        torch.save(self.value_network.state_dict(), os.path.join(base_path, "value_network"))
    
    def load (self, base_path):
        self.value_network.load_state_dict(torch.load(os.path.join(base_path, "value_network")))