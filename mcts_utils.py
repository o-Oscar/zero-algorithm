
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import itertools

import dataclasses

# class MCTS:
#     """
#     Implements the Upper Confidence Polynomial Tree.
#     """
#     def __init__ (self, state, game_model, calc_p, calc_value, gamma):
#         self.state = state
#         self.game_model = game_model
#         self.calc_p = calc_p
#         self.calc_value = calc_value
#         self.gamma = gamma

#         self.next_states = [self.game_model.step(state, i) for i in range(self.game_model.n_action)]
#         self.next_values = [self.calc_value(next_state) for next_state in self.next_states]
#         self.all_rew = [self.game_model.calc_rew(next_state) for next_state in self.next_states]

#         self.cpuct = 4
#         self.all_p = self.calc_p(state) # np.array([self.calc_p(next_state) for next_state in self.next_states])# np.ones((self.game_model.n_action,)) / self.game_model.n_action
#         self.all_q = np.array(self.all_rew) + self.gamma * np.array(self.next_values).flatten()
#         self.all_n = np.zeros((self.game_model.n_action,))
#         self.all_w = np.zeros((self.game_model.n_action,))
#         self.all_u = self.cpuct * self.all_p * np.sqrt(np.sum(self.all_n))/(1+self.all_n)

#         self.children = {}
    
#     def get_action_dist (self):
#         tau = 1
#         n_pow = np.power(self.all_n, 1/tau)
#         return n_pow/np.sum(n_pow)

#     def choose_action (self):
#         action_dist = self.get_action_dist()
#         choosen_action = np.random.choice(self.game_model.n_action, p=action_dist)
#         return choosen_action, action_dist
    
#     def select (self):
#         self.all_u = self.cpuct * self.all_p * np.sqrt(np.sum(self.all_n))/(1+self.all_n)
#         action = np.argmax(self.all_q+self.all_u)
#         if action in self.children:
#             child_q = self.children[action].select()
#         else:
#             child_q = self.expand(action)

#         new_q = self.all_rew[action] + self.gamma * child_q
#         self.new_q = new_q
#         self.all_n[action] = self.all_n[action] + 1
#         self.all_w[action] = self.all_w[action] + new_q
#         self.all_q[action] = self.all_w[action] / self.all_n[action]

#         return new_q
    
#     def expand (self, action):
#         child = MCTS (self.next_states[action], self.game_model, self.calc_p, self.calc_value, self.gamma)
#         self.children[action] = child
#         return np.max(child.all_q)
    
#     def print_hierarchy (self):
#         print(self.state, self.all_n, end="")
#         a = np.argmax(self.all_n)
#         if a in self.children:
#             print(" -> ", end="")
#             self.children[a].print_hierarchy()
#         else:
#             print()
    
# class new_MCTS:
#     """
#     Implements the Upper Confidence Polynomial Tree.
#     """
#     def __init__ (self, state, game_model, calc_p, calc_value, gamma):
#         self.state = state
#         self.game_model = game_model
#         self.calc_p = calc_p
#         self.calc_value = calc_value
#         self.gamma = gamma

#         self.next_states = [self.game_model.step(state, i) for i in range(self.game_model.n_action)]
#         self.next_values = [self.calc_value(next_state) for next_state in self.next_states]
#         self.all_rew = [self.game_model.calc_rew(next_state) for next_state in self.next_states]

#         self.cpuct = 4
#         self.all_p = self.calc_p(state) # np.array([self.calc_p(next_state) for next_state in self.next_states])# np.ones((self.game_model.n_action,)) / self.game_model.n_action
#         self.all_q = np.array(self.all_rew) + self.gamma * np.array(self.next_values).flatten()
#         self.all_n = np.zeros((self.game_model.n_action,))
#         self.all_w = np.zeros((self.game_model.n_action,))
#         self.all_u = self.cpuct * self.all_p * np.sqrt(np.sum(self.all_n))/(1+self.all_n)

#         self.children = {}
    
#     def get_action_dist (self):
#         tau = 1
#         n_pow = np.power(self.all_n, 1/tau)
#         return n_pow/np.sum(n_pow)

#     def choose_action (self):
#         action_dist = self.get_action_dist()
#         choosen_action = np.random.choice(self.game_model.n_action, p=action_dist)
#         return choosen_action, action_dist
    
#     def select (self):
#         self.all_u = self.cpuct * self.all_p * np.sqrt(np.sum(self.all_n))/(1+self.all_n)
#         action = np.argmax(self.all_q+self.all_u)
#         if action in self.children:
#             child_q = self.children[action].select()
#         else:
#             child_q = self.expand(action)

#         new_q = self.all_rew[action] + self.gamma * child_q
#         self.new_q = new_q
#         self.all_n[action] = self.all_n[action] + 1
#         self.all_w[action] = self.all_w[action] + new_q
#         self.all_q[action] = self.all_w[action] / self.all_n[action]

#         return new_q
    
#     def expand (self, action):
#         child = MCTS (self.next_states[action], self.game_model, self.calc_p, self.calc_value, self.gamma)
#         self.children[action] = child
#         return np.max(child.all_q)
    
#     def print_hierarchy (self):
#         print(self.state, self.all_n, end="")
#         a = np.argmax(self.all_n)
#         if a in self.children:
#             print(" -> ", end="")
#             self.children[a].print_hierarchy()
#         else:
#             print()

class MCTS:
    def __init__ (self, state, game_model, calc_prior_p, rollout, calc_value, gamma):
        self.state = state
        self.game_model = game_model
        self.calc_prior_p = calc_prior_p
        self.rollout = rollout
        self.calc_value = calc_value
        self.gamma = gamma

        self.is_leaf = True
        self.children = None

        self.is_recurse = False
        # self.parent_mcts, self.parent_rewards, self.parent_proba = parent_history
        # self.is_recurse = False # np.any([np.all(self.state==test_state) for test_state in self.parent_mcts.state])
        # if self.is_recurse:
        #     # rec_id = self.parent_states.index(self.state)
        #     rec_id = np.min([i for i, test_state in enumerate(self.parent_mcts.state)])
        #     cum_rew = np.sum([self.gamma**(i-rec_id) * self.parent_rewards[i] for i in range(rec_id, len(self.parent_rewards))])
        #     self.value = cum_rew / (1-self.gamma**(len(self.parent_rewards) - rec_id))


    
    def expand (self):

        self.next_states = [self.game_model.step(self.state, i) for i in range(self.game_model.n_action)]
        self.next_values = self.calc_value(self.next_states)
        self.all_rew = [self.game_model.calc_rew(next_state) for next_state in self.next_states]

        self.cpuct = 4
        self.all_p = self.calc_prior_p(self.state) # np.array([self.calc_p(next_state) for next_state in self.next_states])# np.ones((self.game_model.n_action,)) / self.game_model.n_action
        self.all_q = np.array(self.all_rew) + self.gamma * np.array(self.next_values).flatten()
        self.all_n = np.zeros((self.game_model.n_action,))
        self.all_w = np.zeros((self.game_model.n_action,))
        self.all_u = self.cpuct * self.all_p * np.sqrt(np.sum(self.all_n))/(1+self.all_n)

        self.children = [self.init_mcts(next_state) for next_state, rew in zip(self.next_states, self.all_rew)]
        self.is_leaf = False

    def init_mcts (self, new_state):
         return MCTS(new_state, self.game_model, self.calc_prior_p, self.rollout, self.calc_value, self.gamma)

    def select_and_backup (self):
        if self.is_recurse:
            return self.value

        if not self.is_leaf:
            action = np.argmax(self.all_q+self.all_u)
            backup_value = self.all_rew[action] + self.gamma * self.children[action].select_and_backup()
        else:
            action, backup_value = self.rollout (self.state)
            self.expand()
        
        self.all_n[action] = self.all_n[action] + 1
        self.all_w[action] = self.all_w[action] + backup_value
        self.all_q[action] = self.all_w[action] / self.all_n[action]
        self.all_u = self.cpuct * self.all_p * np.sqrt(np.sum(self.all_n))/(1+self.all_n)

        return backup_value

    def get_action_dist (self):
        tau = 1
        n_pow = np.power(self.all_n, 1/tau)
        return n_pow/np.sum(n_pow)

    def choose_action (self):
        action_dist = self.get_action_dist()
        choosen_action = np.random.choice(self.game_model.n_action, p=action_dist)
        return choosen_action, action_dist