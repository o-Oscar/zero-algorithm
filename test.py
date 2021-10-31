
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
        self.fc1 = nn.Linear(inp, 256)
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

class ActionNetwork(nn.Module):

    def __init__(self, inp, out):
        super(ActionNetwork, self).__init__()
        self.fc1 = nn.Linear(inp, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, out)
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
        x = torch.softmax(x, dim=1)
        return x


class GameModel:
    def __init__ (self):
        self.n_pallets = 4
        self.all_actions = [(0,1), (0, 2), (1, 2)]
        
        self.obs_dims = 3*self.n_pallets # TODO : add pillars
        self.n_action = 3 # n_pillar * (n_pillar - 1) / 2

        self.target = [0]*self.n_pallets

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

        self.gamma = 0.97

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

        to_return = {"obs": all_obs, "values": all_values}
        return to_return
    
    def fit (self, train_set):
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

class mctsQLearning (RandQLearning):

    def choose_expl_action (self, state):
        # print(state)
        mcts = MCTS(state, self.game_model, lambda x: self.calc_p(x), lambda next_state: self.value_network(self.game_model.generate_obs(next_state)).detach().numpy(), self.gamma)
        
        for i in range(30):
            # print(mcts.all_n, mcts.all_q, mcts.all_u)
            mcts.select()
            # print(mcts.new_q)
            # mcts.print_hierarchy ()

        return mcts.choose_action()[0]

    def calc_p (self, state):
        return np.ones(self.game_model.n_action)/self.game_model.n_action

class MCTS:
    def __init__ (self, state, game_model, calc_p, calc_value, gamma):
        self.state = state
        self.game_model = game_model
        self.calc_p = calc_p
        self.calc_value = calc_value
        self.gamma = gamma

        self.next_states = [self.game_model.step(state, i) for i in range(self.game_model.n_action)]
        self.next_values = [self.calc_value(next_state) for next_state in self.next_states]
        self.all_rew = [self.game_model.calc_rew(next_state) for next_state in self.next_states]

        self.cpuct = 4
        self.all_p = self.calc_p(state) # np.array([self.calc_p(next_state) for next_state in self.next_states])# np.ones((self.game_model.n_action,)) / self.game_model.n_action
        self.all_q = np.array(self.all_rew) + self.gamma * np.array(self.next_values).flatten()
        self.all_n = np.zeros((self.game_model.n_action,))
        self.all_w = np.zeros((self.game_model.n_action,))
        self.all_u = self.cpuct * self.all_p * np.sqrt(np.sum(self.all_n))/(1+self.all_n)

        self.children = {}
    
    def get_action_dist (self):
        tau = 1
        n_pow = np.power(self.all_n, 1/tau)
        return n_pow/np.sum(n_pow)

    def choose_action (self):
        action_dist = self.get_action_dist()
        choosen_action = np.random.choice(self.game_model.n_action, p=action_dist)
        return choosen_action, action_dist
    
    def select (self):
        self.all_u = self.cpuct * self.all_p * np.sqrt(np.sum(self.all_n))/(1+self.all_n)
        action = np.argmax(self.all_q+self.all_u)
        if action in self.children:
            child_q = self.children[action].select()
        else:
            child_q = self.expand(action)

        new_q = self.all_rew[action] + self.gamma * child_q
        self.new_q = new_q
        self.all_n[action] = self.all_n[action] + 1
        self.all_w[action] = self.all_w[action] + new_q
        self.all_q[action] = self.all_w[action] / self.all_n[action]

        return new_q
    
    def expand (self, action):
        child = MCTS (self.next_states[action], self.game_model, self.calc_p, self.calc_value, self.gamma)
        self.children[action] = child
        return np.max(child.all_q)
    
    def print_hierarchy (self):
        print(self.state, self.all_n, end="")
        a = np.argmax(self.all_n)
        if a in self.children:
            print(" -> ", end="")
            self.children[a].print_hierarchy()
        else:
            print()
    


class mctsActionQLearning (RandQLearning):

    def __init__ (self):
        super().__init__()

        # self.target_action_network = ActionNetwork (self.game_model.obs_dims, self.game_model.n_action)
        self.action_network = ActionNetwork (self.game_model.obs_dims, self.game_model.n_action)


        all_parameters = list(self.value_network.parameters()) + list(self.action_network.parameters())
        self.optimizer = optim.Adam(all_parameters, lr=0.001)


    def choose_expl_action (self, state): # TODO : add the ability to use the actor network in the search process (10 steps of actor-only search)
        mcts = MCTS(state, self.game_model, lambda next_state: self.calc_p (next_state), lambda next_state: self.multi_steps_calc_value(3, next_state), self.gamma)
        
        for i in range(10):
            # print(mcts.all_n, mcts.all_q, mcts.all_u)
            mcts.select()
            # print(mcts.new_q)
            # mcts.print_hierarchy ()
        
        action, action_dist = mcts.choose_action()
        return action, action_dist, mcts.all_q[action]
    
    def multi_steps_calc_value (self, n_steps, state):
        to_return = 0
        fac = 1 # self.gamma
        lamb = 0.
        for step in range(n_steps):
            # test = self.action_network(self.game_model.generate_obs(state)).detach().numpy().flatten()
            # print(test.shape)
            # test = np.ones(self.game_model.n_action)/self.game_model.n_action
            action_dist = (1-lamb) * self.action_network(self.game_model.generate_obs(state)).detach().numpy().flatten() + lamb * np.ones(self.game_model.n_action)/self.game_model.n_action
            # print(np.sum(action_dist))
            action = np.random.choice(self.game_model.n_action, p=action_dist/np.sum(action_dist))
            # action = np.argmax(action_dist)
            # action = np.random.randint(self.game_model.n_action)
            state = self.game_model.step(state, action)
            reward = self.game_model.calc_rew(state)
            to_return += fac * reward
            fac *= self.gamma
        to_return += fac * self.value_network(self.game_model.generate_obs(state)).detach().numpy()
        return to_return
    
    def calc_p (self, next_state):
        lamb = .25#25
        return (1-lamb) * self.action_network(self.game_model.generate_obs(next_state)).detach().numpy().flatten() + lamb * np.ones(self.game_model.n_action)/self.game_model.n_action

    def build_train_set (self, n_sample):
        all_obs = []
        all_values = []
        all_act_dist = []
        for sample in range(n_sample):
            state = self.game_model.reset()
            obs = self.game_model.generate_obs(state)
            action, action_dist, target_value = self.choose_expl_action(state)
            # new_state = self.game_model.step(state, action)
            # new_obs = self.game_model.generate_obs(new_state)
            # rew = self.game_model.calc_rew(new_state)

            all_obs.append(obs)
            all_values.append(target_value)#rew + self.gamma * self.target_network(new_obs))
            all_act_dist.append(action_dist)
        
        all_obs = torch.cat(all_obs, dim=0)
        all_values = torch.tensor(np.stack(all_values))
        all_act_dist = torch.tensor(np.stack(all_act_dist, axis=0))

        to_return = {"obs": all_obs, "values": all_values, "act_dist": all_act_dist}
        return to_return
    
    
    def fit (self, train_set):
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

            action_loss = -torch.mean(torch.sum((all_act_dist + init_act_dist*0.1) * torch.log(self.action_network(all_obs)), dim=1))
            value_loss = torch.mean(torch.square(cur_value - all_values) + torch.square(cur_value-init_value) / self.value_epsilon)
            
            loss = action_loss + value_loss
            loss.backward()
            all_loss.append(loss.detach().numpy())
            all_actor_loss.append(action_loss.detach().numpy())
            self.optimizer.step()
        return all_actor_loss

    
def main ():
    # print(torch.cuda.is_available())
    # qLearning = EpsilonQLearning()
    # qLearning = RandQLearning ()
    # qLearning = onestepQLearning ()
    # qLearning = mctsQLearning()
    qLearning = mctsActionQLearning()

    # state = [1, 1]
    # action = 0
    # n_state = qLeraning.game_model.step(state, action)
    # print(n_state)
    # print(qLeraning.game_model.generate_obs(state))
    # print(qLeraning.game_model.generate_obs(n_state))

    # all_obs, all_values = qLeraning.build_train_set(10)
    # print(all_obs)
    # print(all_values)


    # all_test_states = [np.array([k, j, i]) for i, j, k in itertools.product(range(3), range(3), range(3))]
    all_test_states = [np.array([l, k, j, i]) for i, j, k, l in itertools.product(range(3), range(3), range(3), range(3))]
    all_test_obs = torch.cat([qLearning.game_model.generate_obs(state) for state in all_test_states], dim=0)

    # print(qLearning.choose_expl_action(all_test_states[3]))
    # exit()

    # for state in all_test_states:
    #     print(qLearning.game_model.step(state, 0))
    #     print(qLearning.game_model.calc_rew(state))
    #     print(qLearning.choose_expl_action(state))

    #     print()
        
    # inp = torch.tensor(np.array([[1, 0, 0], [1, 0, 0]]).reshape((1,6)).astype(np.float32))
    # print(all_test_obs)
    # print(qLearning.value_network(all_test_obs))
    # print(qLeraning.value_network(inp))

    if (True):
        to_plot = []
        to_plot.append(qLearning.value_network(all_test_obs).detach().numpy())
        to_plot2 = []
        to_plot3 = []
        n_epoch = 100
        for i in range(n_epoch):
            train_set = qLearning.build_train_set(10)
            all_obs, all_values = train_set["obs"], train_set["values"]
            cur_plot = []
            for state in all_test_states:
                to_print = [float(value.detach().numpy()) for obs, value in zip(all_obs, all_values) if np.all(obs.detach().numpy() == qLearning.game_model.generate_obs(state).detach().numpy())]
                cur_plot.append(np.mean(to_print))
            to_plot2.append(cur_plot)
            # print(all_values)
            qLearning.value_epsilon = np.power(10, -i/n_epoch)
            all_action_loss = qLearning.fit (train_set)
            qLearning.fit_traget_network ()
            # print(qLearning.value_network(inp))
            # print(qLearning.value_network(all_test_obs))
            to_plot.append(qLearning.value_network(all_test_obs).detach().numpy())
            # print(qLearning.choose_expl_action(all_test_states[0]))
            if all_action_loss is not None:
                to_plot3 = to_plot3 + all_action_loss
                # qLearning.evaluate_learned_policy (all_test_states)
        plt.plot(to_plot3)
        plt.show()

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