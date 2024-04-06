import itertools

import joblib
import numpy as np

from non_linear_solver import LagrangeSolver, NonLPSolver


class play_MinMax_CombSA:
    def __init__(
        self,
        actions_list,
        values,
        budget,
        R_sub_Gaussian,
        allocatin_vector_type,
        beta,
    ) -> None:
        self.actions_list = actions_list
        self.K_actions = len(self.actions_list)
        self.values = values
        self.budget = budget
        self.R_sub_Gaussian = R_sub_Gaussian
        self.beta = beta
        self.N = len(values)
        self.allocation_vector_type = allocatin_vector_type
        # 実験を通して更新されていくもの
        self.sample_mean_values = [0 for row in range(self.N)]
        self.num_pulls = [0 for row in range(self.N)]
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
        # Initialization
        # スパース性に対応。
        INITIALIZATION_ROUND = int(
            np.ceil(
                (self.budget - self.N * np.ceil(np.log2(self.K_actions)))
                / self.N
                * self.beta
            )
        )
        for _ in range(INITIALIZATION_ROUND):
            for i in range(self.N):
                observation = self.sample_edge(self.values[i])
                self.sample_mean_values[i] = (
                    self.sample_mean_values[i] * self.num_pulls[i] + observation
                ) / (self.num_pulls[i] + 1)
                self.num_pulls[i] += 1

        # Loop
        Total_rounds = np.int(np.ceil(np.log2(self.K_actions)))
        B = 2 ** (np.ceil(np.log2(self.K_actions))) - 1
        T_dash = self.budget - INITIALIZATION_ROUND * self.N
        for r in range(1, Total_rounds + 1):
            m_r = (T_dash - self.N * np.ceil(np.log2(self.K_actions))) / (
                B / (2 ** (r - 1))
            )
            n_r = None
            combination_list = []
            for i in range(len(self.actions_list) - 1):
                for j in range(i + 1, len(self.actions_list)):
                    if i != j:
                        combination_list.append((i, j))

            for_loop_result = None
            if self.allocation_vector_type == "non_linear_solver":
                for_loop_result = joblib.Parallel(n_jobs=-1)(
                    joblib.delayed(NonLPSolver)(
                        X=np.array(
                            (
                                np.array(self.actions_list[i])
                                - np.array(self.actions_list[j])
                            )
                            ** 2
                        ),
                        T=np.array(self.num_pulls),
                        m_r=m_r,
                    )
                    for (i, j) in combination_list
                )
            if self.allocation_vector_type == "Lagrange":
                for_loop_result = joblib.Parallel(n_jobs=-1)(
                    joblib.delayed(LagrangeSolver)(
                        X=np.array(
                            (
                                np.array(self.actions_list[i])
                                - np.array(self.actions_list[j])
                            )
                            ** 2
                        ),
                        T=np.array(self.num_pulls),
                        m_r=m_r,
                    )
                    for (i, j) in combination_list
                )

            for_loop_result = sorted(for_loop_result, key=lambda x: x[1])
            n_r = np.ceil(for_loop_result[-1][0])

            # pull arms according to list n_r
            for i in range(self.N):
                for _ in range(int(n_r[i])):
                    observation = self.sample_edge(self.values[i])
                    self.sample_mean_values[i] = (
                        self.sample_mean_values[i] * self.num_pulls[i] + observation
                    ) / (self.num_pulls[i] + 1)
                    self.num_pulls[i] += 1
            # sort actions in the current actions_list
            for_sort_list = [
                [self.actions_list[i], 0] for i in range(len(self.actions_list))
            ]
            for i in range(len(self.actions_list)):
                for_sort_list[i][1] = np.dot(
                    np.array(for_sort_list[i][0]), np.array(self.sample_mean_values)
                )
            for_sort_list = sorted(for_sort_list, key=lambda x: x[1])
            self.actions_list = [for_sort_list[i][0] for i in range(len(for_sort_list))]
            self.actions_list.reverse()
            self.actions_list = self.actions_list[
                : np.int(np.ceil(self.K_actions / (2**r)))
            ]

        if len(self.actions_list) > 1:
            exit()
        print("sum of num_pulls", np.sum(self.num_pulls))
        return self.actions_list[0], self.optimal_pi
