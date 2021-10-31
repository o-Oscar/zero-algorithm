
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

import itertools

from zeroAlgo.randQLeraning import RandQLearning
from zeroAlgo.epsilonQLearning import EpsilonQLearning
from zeroAlgo.mctsQLearning import MctsQLearning
from zeroAlgo.mctsActionQLearning import mctsActionQLearning

from game.hanoi import HanoiTower

    
def main ():
    # print(torch.cuda.is_available())

    game_model = HanoiTower(n_pallets=5)

    # qLearning = RandQLearning (game_model)
    # qLearning = EpsilonQLearning(game_model)
    # qLearning = MctsQLearning(game_model)
    qLearning = mctsActionQLearning(game_model)

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
    all_test_states = game_model.get_all_states() # [np.array([l, k, j, i]) for i, j, k, l in itertools.product(range(3), range(3), range(3), range(3))]
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
            qLearning.n_action_only = 10
            if qLearning.n_action_only < 1:
                qLearning.n_action_only = 1
            if qLearning.n_action_only > 10:
                qLearning.n_action_only = 10

            train_set = qLearning.build_train_set(30)
            all_obs, all_values = train_set["obs"], train_set["values"]
            cur_plot = []
            for state in all_test_states:
                to_print = [float(value.detach().numpy()) for obs, value in zip(all_obs, all_values) if np.all(obs.detach().numpy() == qLearning.game_model.generate_obs(state).detach().numpy())]
                cur_plot.append(np.mean(to_print))
            to_plot2.append(cur_plot)
            # print(all_values)
            qLearning.value_epsilon = np.power(10, -i/n_epoch)
            all_action_loss = qLearning.fit (train_set)
            if all_action_loss is None:
                qLearning.fit_traget_network ()
            # print(qLearning.value_network(inp))
            # print(qLearning.value_network(all_test_obs))
            to_plot.append(qLearning.value_network(all_test_obs).detach().numpy())
            # print(qLearning.choose_expl_action(all_test_states[0]))
            if all_action_loss is not None:
                to_plot3 = to_plot3 + all_action_loss
                # qLearning.evaluate_learned_policy (all_test_states)
        if all_action_loss is not None:
            plt.plot(to_plot3)
            plt.show()

        to_plot = np.stack(to_plot)
        to_plot2 = np.array(to_plot2)
        plt.plot(to_plot, "-")
        plt.show()
        plt.plot(to_plot2, "o")
        plt.show()

    # print(all_test_obs)

    # for state in all_test_states:
    #     res = qLearning.choose_best_action(state)
    #     print(res)
    #     print(qLearning.game_model.calc_rew(state))
    #     print()



if __name__ == "__main__":
    main ()