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


class play_CSA:
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
        self.N = len(values)
        # 実験を通して更新されていくもの
        self.sample_mean_values = [0 for row in range(self.N)]
        self.num_pulls = [0 for row in range(self.N)]
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

    def tilde_log(self, n):
        ret = 0
        for i in range(1, n + 1):
            ret += 1 / i
        return ret

    def COracle(self, S_t):
        current, pi = -float("inf"), None
        if self.actionset_size == "poly":
            for action in self.actions_list:
                valid_flag = True
                for e, x_e in S_t:
                    if action[e] != x_e:
                        valid_flag = False
                        break
                if valid_flag:
                    compare = np.dot(
                        np.array(action), np.array(self.sample_mean_values)
                    )
                    if compare > current:
                        current = compare
                        pi = action
        elif self.actionset_size == "exp":
            rewards = copy.deepcopy(self.sample_mean_values)
            total_weight = self.total_weight
            for e, x_e in S_t:
                rewards[e] = 0
                total_weight -= self.weights[e] * x_e
            if total_weight >= 0:
                pi = PsuedoPolynomialOracle(total_weight, rewards, self.weights)
                for e, x_e in S_t:
                    pi[e] = x_e

        return pi

    def T_tilde(self, t):
        if t == 0:
            return 0
        tilde_log = self.tilde_log(self.N)
        return int(np.ceil((self.budget - self.N) / (tilde_log * (self.N - t + 1))))

    def play(self):
        F_t, S_t = {}, []

        for t in range(1, self.N + 1):
            prev = self.T_tilde(t - 1)
            tilde_T = self.T_tilde(t) - prev
            for i in range(self.N):
                if i in F_t:
                    continue
                for _ in range(tilde_T):
                    observation = self.sample_edge(self.values[i])
                    self.sample_mean_values[i] = (
                        self.sample_mean_values[i] * self.num_pulls[i] + observation
                    ) / (self.num_pulls[i] + 1)
                    self.num_pulls[i] += 1
            pi_t = self.COracle(S_t)
            if pi_t is None:
                break

            compare_list = []
            for i in range(self.N):
                if i in F_t:
                    continue
                # Search \Tilde{\pi}^{e}
                tilde_pi_te = None
                for x_e in range(self.total_weight // self.weights[i] + 1):
                    if pi_t[i] == x_e:
                        continue
                    # Candidate of \Tilde{\pi}^{e}
                    tilde_pi_t_xe = self.COracle(S_t + [(i, x_e)])
                    if tilde_pi_t_xe is None:
                        continue
                    if tilde_pi_te is None:
                        tilde_pi_te = tilde_pi_t_xe
                    elif np.dot(
                        np.array(tilde_pi_t_xe), np.array(self.sample_mean_values)
                    ) > np.dot(
                        np.array(tilde_pi_te), np.array(self.sample_mean_values)
                    ):
                        tilde_pi_te = tilde_pi_t_xe
                if tilde_pi_te is not None:
                    compare_list.append(
                        [
                            (
                                np.dot(
                                    self.sample_mean_values,
                                    np.array(pi_t) - np.array(tilde_pi_te),
                                )
                            )
                            / (np.abs(pi_t[i] - tilde_pi_te[i])),
                            (i, pi_t[i]),
                        ]
                    )
                elif tilde_pi_te is None:
                    compare_list.append(
                        [
                            (10**6),
                            (i, pi_t[i]),
                        ]
                    )
            compare_list.sort()
            compare_list.reverse()

            F_t[compare_list[0][1][0]] = True
            S_t.append((compare_list[0][1]))
        answer_pi = np.array([-1 for j in range(self.N)])
        for e, x_e in S_t:
            answer_pi[e] = x_e
        print("num_pulls", np.sum(self.num_pulls))
        return answer_pi, self.optimal_pi
