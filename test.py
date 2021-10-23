
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import itertools


class ValueNetwork(nn.Module):

    def __init__(self, inp):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(inp, 256)  # 5*5 from image dimension
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        state_dict = self.fc3.state_dict()
        state_dict["bias"] = (state_dict["bias"]*0).detach()
        state_dict["weight"] = (state_dict["weight"]*0).detach()
        self.fc3.load_state_dict(state_dict)
        # print([x for x in self.fc3.load_state_dict()])
        # print(state_dict)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.squeeze(x, 1)
        return x


class GameModel:
    def __init__ (self):
        self.n_pallets = 2
        self.all_actions = [(0,1), (0, 2), (1, 2)]
        
        self.obs_dims = 3*self.n_pallets # TODO : add pillars
        self.n_action = 3 # n_pillar * (n_pillar - 1) / 2

        self.target = ([0, 0])

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


class RandQLearning:
    def __init__ (self):
        self.game_model = GameModel()
        self.cur_state = self.game_model.reset()

        self.value_network = ValueNetwork(self.game_model.obs_dims)
        self.target_network = ValueNetwork(self.game_model.obs_dims)
        self.target_network.load_state_dict(self.value_network.state_dict())

        self.optimizer = optim.Adam(self.value_network.parameters(), lr=0.001)

        self.gamma = 0.9

        self.value_epsilon = 0.01

    def choose_expl_action (self, state):
        return np.random.randint(self.game_model.n_action)

    def choose_best_action (self, state):
        # print([self.game_model.generate_obs(self.game_model.step(state, i)) for i in range(self.game_model.n_action)])
        next_states = [self.game_model.step(state, i) for i in range(self.game_model.n_action)]
        all_next_values = [self.value_network(self.game_model.generate_obs(next_state)).detach().numpy() for next_state in next_states]
        all_rew = [self.game_model.calc_rew(next_state) for next_state in next_states]
        # print(all_values)
        action = np.argmax([rew + self.gamma * value for rew, value in zip(all_rew, all_next_values)])
        return action

    def calc_value (self, state):
        obs = self.game_model.generate_obs (state)
        return self.value_network(obs)
    
    def build_train_set (self, n_sample):
        all_obs = []
        all_values = []
        for sample in range(n_sample):
            state = self.game_model.reset()
            obs = self.game_model.generate_obs(state)
            action = self.choose_expl_action(state)
            new_state = self.game_model.step(state, action)
            new_obs = self.game_model.generate_obs(new_state)
            rew = self.game_model.calc_rew(new_state)
            # rew = self.game_model.calc_rew(state)

            all_obs.append(obs)
            all_values.append(rew + self.gamma * self.target_network(new_obs))
        
        all_obs = torch.cat(all_obs, dim=0)
        all_values = torch.cat(all_values, dim=0).detach()

        return all_obs, all_values
    
    def fit (self, all_obs, all_values):
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
        # plt.plot(all_loss)
        # plt.show()
    def fit_traget_network (self):
        retention = 0.2
        new_target_dict = {}
        for (key, cur_value), (_, target_value) in zip(self.value_network.state_dict().items(), self.target_network.state_dict().items()):
            new_target_dict[key] = retention * target_value + (1-retention)* cur_value
        self.target_network.load_state_dict(new_target_dict)

class EpsilonQLearning (RandQLearning):
    def choose_expl_action (self, state):
        if np.random.random() < 0.1:
            return np.random.randint(self.game_model.n_action)
        else:
            return self.choose_best_action(state)



def main ():
    # print(torch.cuda.is_available())
    qLearning = EpsilonQLearning()
    # qLearning = RandQLearning ()

    # state = [1, 1]
    # action = 0
    # n_state = qLeraning.game_model.step(state, action)
    # print(n_state)
    # print(qLeraning.game_model.generate_obs(state))
    # print(qLeraning.game_model.generate_obs(n_state))

    # all_obs, all_values = qLeraning.build_train_set(10)
    # print(all_obs)
    # print(all_values)


    all_test_states = [np.array([j, i]) for i, j in itertools.product(range(3), range(3))]
    all_test_obs = torch.cat([qLearning.game_model.generate_obs(state) for state in all_test_states], dim=0)

    # for state in all_test_states:
    #     print(qLearning.game_model.step(state, 0))
    #     print(qLearning.game_model.calc_rew(state))
    #     print(qLearning.choose_expl_action(state))

    #     print()
        
    # inp = torch.tensor(np.array([[1, 0, 0], [1, 0, 0]]).reshape((1,6)).astype(np.float32))
    # print(all_test_obs)
    # print(qLearning.value_network(all_test_obs))
    # print(qLeraning.value_network(inp))

    to_plot = []
    to_plot.append(qLearning.value_network(all_test_obs).detach().numpy())
    to_plot2 = []
    n_epoch = 300
    for i in range(n_epoch):
        all_obs, all_values = qLearning.build_train_set(3)
        cur_plot = []
        for state in all_test_states:
            to_print = [float(value.detach().numpy()) for obs, value in zip(all_obs, all_values) if np.all(obs.detach().numpy() == qLearning.game_model.generate_obs(state).detach().numpy())]
            cur_plot.append(np.mean(to_print))
        to_plot2.append(cur_plot)
        # print(all_values)
        qLearning.value_epsilon = np.power(10, -i/n_epoch)
        qLearning.fit (all_obs, all_values)
        qLearning.fit_traget_network ()
        # print(qLearning.value_network(inp))
        print(qLearning.value_network(all_test_obs))
        to_plot.append(qLearning.value_network(all_test_obs).detach().numpy())

    to_plot = np.stack(to_plot)
    to_plot2 = np.array(to_plot2)
    plt.plot(to_plot)
    plt.show()
    plt.plot(to_plot2)
    plt.show()

    # print(all_test_obs)

    # for state in all_test_states:
    #     res = qLearning.choose_best_action(state)
    #     print(res)
    #     print(qLearning.game_model.calc_rew(state))
    #     print()



if __name__ == "__main__":
    main ()