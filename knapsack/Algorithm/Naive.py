import copy
from collections import defaultdict

import numpy as np


def PsuedoPolynomialOracle(total_weight, reward_values, weights):
    d = len(reward_values)
    ret_action = [0 for _ in range(d)]
    dp = [[-1, defaultdict(int)] for _ in range(total_weight + 1)]
    # 何も入れない時は価値0
    dp[0][0] = 0
    for loop in range(d):
        value, weight = reward_values[loop], weights[loop]
        for i in range(weight, total_weight + 1):
            if dp[i - weight][0] == -1:
                pass
            else:
                if dp[i][0] < dp[i - weight][0] + value:
                    dp[i][0] = dp[i - weight][0] + value
                    dp[i][1] = copy.deepcopy(dp[i - weight][1])
                    dp[i][1][loop] += 1

    current = -float("inf")
    for i in range(total_weight + 1):
        if dp[i][0] > current:
            current = dp[i][0]
            ans = dp[i][1]
    for key in ans:
        ret_action[key] = ans[key]
    return ret_action


class play_Naive:
    def __init__(
        self,
        actionset_size,
        actions_list,
        total_weight,
        values,
        weights,
        budget,
        R_sub_Gaussian,
    ):
        self.actionset_size = actionset_size
        self.total_weight = total_weight
        self.values = values
        self.weights = weights
        self.budget = budget
        self.R_sub_Gaussian = R_sub_Gaussian
        self.d = len(values)
        # 実験を通して更新されていくもの
        self.sample_mean_values = [0 for row in range(self.d)]
        self.num_pulls = [0 for row in range(self.d)]
        if self.actionset_size == "poly":
            self.actions_list = actions_list
            self.K_actions = len(self.actions_list)
            # optimal actionを保持する
            true_reward_list = [
                [np.dot(np.array(self.actions_list[i]), np.array(self.values)), i]
                for i in range(self.K_actions)
            ]
            true_reward_list.sort()
            if self.actionset_size == "poly":
                self.optimal_pi = self.actions_list[true_reward_list[-1][1]]
        elif self.actionset_size == "exp":
            self.optimal_pi = PsuedoPolynomialOracle(
                self.total_weight, self.values, self.weights
            )

    def sample_edge(self, mean):
        sample_result = np.random.normal(mean, self.R_sub_Gaussian, 1)[0]
        return sample_result

    def play(self):
        pull_num = self.budget // self.d
        for i in range(self.d):
            for _ in range(pull_num):
                observation = self.sample_edge(self.values[i])
                self.sample_mean_values[i] = (
                    self.sample_mean_values[i] * self.num_pulls[i] + observation
                ) / (self.num_pulls[i] + 1)
                self.num_pulls[i] += 1
        ret = PsuedoPolynomialOracle(
            self.total_weight, self.sample_mean_values, self.weights
        )
        return ret, self.optimal_pi
