import copy
from itertools import combinations, product

import numpy as np

from PEACE_ALG.transductive_bandits_alg import _rounding, gamma_tb


class play_PEACE:
    def __init__(
        self, actions_list, values, budget, R_sub_Gaussian, epsilon=1 / 10, delta=0.1
    ):
        self.actions_list = actions_list
        self.K_actions = len(self.actions_list)
        self.values = values
        self.budget = budget
        self.R_sub_Gaussian = R_sub_Gaussian
        self.N = len(values)
        self.epsilon = epsilon
        self.delta = delta
        # 実験を通して更新されていくもの
        self.sample_mean_values = [0 for _ in range(self.N)]
        self.num_pulls = [0 for _ in range(self.N)]
        # optimal actionを保持する
        true_reward_list = [
            [np.dot(np.array(self.actions_list[i]), np.array(self.values)), i]
            for i in range(self.K_actions)
        ]
        true_reward_list.sort()
        self.optimal_pi = self.actions_list[true_reward_list[-1][1]]

    def sample_edge(self, mean):
        sample_result = np.random.normal(mean, self.R_sub_Gaussian, 1)[0]
        return sample_result

    def play(
        self,
    ):
        iters = 50
        X = np.eye(self.N)
        mathcal_Z = copy.deepcopy(self.actions_list)
        _, initial_gamma = gamma_tb(mathcal_Z, iters=iters)
        num_epochs = int(np.ceil(max(np.log2(initial_gamma), 2)))
        epoch_length = int(np.floor(self.budget / num_epochs))

        Z_k = mathcal_Z

        for epoch in range(num_epochs):
            print(epoch, num_epochs)
            lambda_k, gamma_k = gamma_tb(np.array(Z_k), iters=iters)
            allocation = _rounding(lambda_k, epoch_length)
            for s, num in enumerate(allocation):
                for _ in range(num):
                    observation = self.sample_edge(self.values[s])
                    self.sample_mean_values[s] = (
                        self.sample_mean_values[s] * self.num_pulls[s] + observation
                    ) / (self.num_pulls[s] + 1)
                    self.num_pulls[s] += 1
            if len(Z_k) > 2:
                sort_list = []
                for z in Z_k:
                    sort_list.append(
                        [np.dot(np.array(z), np.array(self.sample_mean_values)), z]
                    )
                sort_list.sort()
                sort_list.reverse()
                new_Z_k = []
                for _, z in sort_list:
                    new_Z_k.append(z)
                    _, gamma = gamma_tb(np.array(new_Z_k), iters=iters)
                    if gamma > gamma_k / 2:
                        break
                    Z_k = new_Z_k
            if len(Z_k) == 1:
                break
            
        print("sum of num_pulls is ", np.sum(self.num_pulls))
        max_index = np.argmax(np.array(Z_k) @ np.array(self.sample_mean_values))
        return Z_k[max_index], self.optimal_pi
