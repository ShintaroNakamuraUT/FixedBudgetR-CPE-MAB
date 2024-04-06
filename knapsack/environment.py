import copy
from collections import defaultdict

import numpy as np


class actions_list_maker:
    def __init__(self, weights, total_weights, value_width, K_actions, value_param):
        self.weights = weights
        self.total_weights = total_weights
        self.value_width = value_width
        self.K_actions = K_actions
        self.value_param = value_param
        self.actions_list = []
        self.N = len(weights)

    def solver(self):
        for num_iter in range(1, self.K_actions):
            if len(self.actions_list) == self.K_actions:
                break
            ret_action = [0 for _ in range(self.N)]
            dp = [[-1, defaultdict(int)] for _ in range(self.total_weights + 1)]
            # 何も入れない時は価値0
            dp[0][0] = 0
            # dummy_values = np.random.normal(1, 1, self.N)
            dummy_values = self.weights * (
                1 + self.value_param * np.random.rand(self.N)
            )
            # dummy_values = self.weights + (
            #     -self.value_width + self.value_width * np.random.rand(self.N)
            # )
            for item_index in range(self.N):
                value, weight = dummy_values[item_index], self.weights[item_index]
                for i in range(weight, self.total_weights + 1):
                    if dp[i - weight][0] == -1:
                        pass
                    else:
                        if dp[i][0] < dp[i - weight][0] + value:
                            dp[i][0] = dp[i - weight][0] + value
                            dp[i][1] = copy.deepcopy(dp[i - weight][1])
                            dp[i][1][item_index] += 1
            current = -float("inf")
            for i in range(self.total_weights + 1):
                if dp[i][0] > current:
                    current = dp[i][0]
                    ans = dp[i][1]
            for key in ans:
                ret_action[key] = ans[key]
            if ret_action not in self.actions_list:
                self.actions_list.append(ret_action)
        print("K = ", len(self.actions_list))
