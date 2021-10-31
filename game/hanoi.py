
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import itertools

class HanoiTower:
    def __init__ (self, n_pallets=3):
        self.n_pallets = n_pallets
        self.all_actions = [(0,1), (0, 2), (1, 2)]
        
        
        self.obs_dims = 3*self.n_pallets # TODO : add pillars
        self.n_action = 3 # n_pillar * (n_pillar - 1) / 2

        self.target = [0]*self.n_pallets

        self.gamma = 0.97
    
    def get_all_states (self):
        ranges = [range(3) for i in range(self.n_pallets)]
        to_return = [np.array(x) for x in itertools.product(*ranges)]
        return to_return

    def reset (self):
        return np.random.randint(3, size=(self.n_pallets, ))
    
    def step (self, state, action):
        # if np.all(state == self.target):
        #     return state[:]

        int_action = int(action)
        f_stack, s_stack = self.all_actions[int_action]
        f_size = min([self.n_pallets] + [i for i in range(self.n_pallets) if state[i] == f_stack])
        s_size = min([self.n_pallets] + [i for i in range(self.n_pallets) if state[i] == s_stack])
        if f_size > s_size:
            f_stack, s_stack = (s_stack, f_stack)
            f_size, s_size = (s_size, f_size)
        
        to_return = [x for x in state]
        if not f_size == self.n_pallets:
            to_return[f_size] = s_stack
        else:
            # print("no_move")
            pass
        return to_return
    
    def generate_obs (self, state):
        to_return = np.zeros((self.n_pallets, 3))
        for i in range(self.n_pallets):
            to_return[i, state[i]] = 1
        
        obs = torch.tensor(to_return.reshape((1, self.obs_dims,)).astype(np.float32))
        return obs
    
    def calc_rew (self, state):
        return 1 if np.all(state == self.target) else 0 # we can put some reward shaping to better simulate a robotics setup
