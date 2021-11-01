
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import time
import os
import itertools

from mcts_utils import MCTS
from networks import ValueNetwork, ActionNetwork


class mctsActionQLearning ():
    """
    This algorithm trains both a value network and an action network to try and solve a game. 
    The value network perdicts the value of a specified state, while the action network predict a distribution over the action space for this same state.
    To choose an action, we use MCTS search.
    """

    def __init__ (self, game_model):
        self.game_model = game_model

        self.gamma = self.game_model.gamma

        self.value_epsilon = 0.01

        self.value_network = ValueNetwork(self.game_model.obs_dims)
        self.action_network = ActionNetwork (self.game_model.obs_dims, self.game_model.n_action)

        all_parameters = list(self.value_network.parameters()) + list(self.action_network.parameters())
        self.optimizer = optim.Adam(all_parameters, lr=0.001)

        self.n_action_only = 1


    def set_train_advancement (self, adv):
        self.n_action_only = 1 if adv < .3 else 10
        self.value_epsilon = np.power(10, 10-10*adv)

    def choose_expl_action (self, state):
        """
        We use a Monte-Carlo Tree Search (MCTS) to compute a distribution over the action space.
        During this search, prior q-value is given by the value network, and prior action distribution is given by the action network.
        The leaf value of the MCTS is computed using a rollout of the game that only uses the action distribution provided by the action network, with termination value given by the value network.
        """
        start = time.time()
        mcts = MCTS(state, self.game_model, 
                lambda next_state: self.calc_p (next_state), 
                lambda next_state: self.multi_steps_calc_value(self.n_action_only, next_state), 
                lambda next_states: self.value_network(torch.tensor(np.stack([self.game_model.generate_obs(next_state) for next_state in next_states]))).detach().numpy(),
                self.gamma)
        
        for i in range(30):
            # print(mcts.all_n, mcts.all_q, mcts.all_u)
            mcts.select_and_backup()
            # print(mcts.new_q)
            # mcts.print_hierarchy ()
        
        action, action_dist = mcts.choose_action()
        # print(time.time()-start)
        return action, action_dist, mcts.all_q[action]
    
    def choose_best_action (self, state):
        # action, action_dist, q_value = self.choose_expl_action (state)
        action = np.argmax(self.action_network(self.game_model.generate_obs(state)).detach().numpy().flatten())
        return action

    def multi_steps_calc_value (self, n_steps, state):
        to_return = 0
        fac = 1 # self.gamma
        lamb = 0.
        for step in range(n_steps):
            action_dist = (1-lamb) * self.action_network(self.game_model.generate_obs(state)).detach().numpy().flatten() + lamb * np.ones(self.game_model.n_action)/self.game_model.n_action
            action = np.random.choice(self.game_model.n_action, p=action_dist/np.sum(action_dist))
            if step == 0:
                first_action = action
            # action = np.argmax(action_dist)
            # action = np.random.randint(self.game_model.n_action)
            state = self.game_model.step(state, action)
            reward = self.game_model.calc_rew(state)
            to_return += fac * reward
            fac *= self.gamma
        to_return += fac * self.value_network(self.game_model.generate_obs(state)).detach().numpy()
        return first_action, to_return
    
    def calc_p (self, next_state):
        """
        The prior over the action space is a sum of a uniform and the prediction of the action network.
        """
        lamb = .25#25
        return (1-lamb) * self.action_network(self.game_model.generate_obs(next_state)).detach().numpy().flatten() + lamb * np.ones(self.game_model.n_action)/self.game_model.n_action

    def build_train_set (self, n_sample):
        """
        The training set is composed of triplets state-value-action_dist. 
        The states are sampled uniformly over the state space.
        Both the value target and the action distribution target are computed using the MCTS. 
        """
        all_obs = []
        all_values = []
        all_act_dist = []
        for sample in range(n_sample):
            if sample % 10 == 0:
                state = self.game_model.reset()
            else:
                state = self.game_model.step(state, action)

            obs = self.game_model.generate_obs(state)
            action, action_dist, target_value = self.choose_expl_action(state)

            all_obs.append(obs)
            all_values.append(target_value)#rew + self.gamma * self.target_network(new_obs))
            all_act_dist.append(action_dist)
        
        all_obs = torch.cat(all_obs, dim=0)
        all_values = torch.tensor(np.stack(all_values))
        all_act_dist = torch.tensor(np.stack(all_act_dist, axis=0))

        to_return = {"obs": all_obs, "values": all_values, "act_dist": all_act_dist}
        return to_return
    
    
    def fit (self, train_set):
        """
        Fits the value and action network to the dataset previously generated.
        """
        all_obs = train_set["obs"]
        all_values = train_set["values"]
        all_act_dist = train_set["act_dist"]
        n_sub_epoch = 100
        all_actor_loss = []
        all_loss = []

        init_value = self.value_network(all_obs).detach()
        init_act_dist = self.action_network(all_obs).detach()


        for epoch in range(n_sub_epoch):
            self.optimizer.zero_grad()
            
            cur_value = self.value_network(all_obs)

            targ_dist = all_act_dist + init_act_dist*0.1
            targ_dist = targ_dist / torch.sum(targ_dist, dim=1, keepdim=True)

            action_loss = -torch.mean(torch.sum(targ_dist * torch.log(self.action_network(all_obs)), dim=1)) + torch.mean(torch.sum(targ_dist * torch.log(targ_dist), dim=1))
            value_loss = torch.mean(torch.square(cur_value - all_values) + torch.square(cur_value-init_value) / self.value_epsilon)
            
            loss = action_loss + value_loss
            loss.backward()
            all_loss.append(loss.detach().numpy())
            all_actor_loss.append(action_loss.detach().numpy())
            self.optimizer.step()
        return all_actor_loss

    def save (self, base_path):
        torch.save(self.value_network.state_dict(), os.path.join(base_path, "value_network"))
        torch.save(self.action_network.state_dict(), os.path.join(base_path, "action_network"))

    
    def load (self, base_path):
        self.value_network.load_state_dict(torch.load(os.path.join(base_path, "value_network")))
        self.action_network.load_state_dict(torch.load(os.path.join(base_path, "action_network")))