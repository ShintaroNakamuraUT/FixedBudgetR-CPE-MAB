import joblib
import numpy as np

from non_linear_solver import LagrangeSolver


class play_MinMax_CombSA_new:
    def __init__(
        self,
        actions_list,
        supply_vector,
        demand_vector,
        cost_matrix,
        budget,
        R_sub_Gaussian,
        allocation_vector_type,
        beta,
    ):
        self.actions_list = actions_list
        self.supply_vector = supply_vector
        self.demand_vector = demand_vector
        self.cost_matrix = cost_matrix.flatten()  # We convert cost matrix to 1D vector
        self.d = len(self.cost_matrix)
        self.budget = budget
        self.R_sub_Gaussian = R_sub_Gaussian
        self.allocation_vector_type = allocation_vector_type
        self.beta = beta

        # The number of actions
        self.K = len(self.actions_list)

        self.sample_mean_values = [0 for row in range(self.d)]
        self.num_pulls = [0 for row in range(self.d)]

        # Compute the optimal action
        true_reward_list = [
            [np.dot(np.array(self.actions_list[i]), np.array(self.cost_matrix)), i]
            for i in range(self.K)
        ]
        true_reward_list.sort()

        self.optimal_pi = self.actions_list[true_reward_list[-1][1]]

    def sample_edge(self, mean):
        sample_result = np.random.normal(mean, self.R_sub_Gaussian, 1)[0]
        return sample_result

    def play(
        self,
    ):
        INITIALIZATION_ROUND = int(np.floor(self.budget / self.d * self.beta))
        for _ in range(INITIALIZATION_ROUND):
            for i in range(self.d):
                observation = self.sample_edge(self.cost_matrix[i])
                self.sample_mean_values[i] = (
                    self.sample_mean_values[i] * self.num_pulls[i] + observation
                ) / (self.num_pulls[i] + 1)
                self.num_pulls[i] += 1

        # Loop
        Total_rounds = int(np.ceil(np.log2(self.d)))
        B = 2 ** (np.ceil(np.log2(self.d))) - 1
        T_dash = self.budget - INITIALIZATION_ROUND * self.d

        for r in range(1, Total_rounds + 1):
            m_r = (T_dash - self.d * np.ceil(np.log2(self.d))) / (B / 2 ** (r - 1))
            n_r = None
            combination_list = []
            for i in range(len(self.actions_list) - 1):
                for j in range(i + 1, len(self.actions_list)):
                    if i != j:
                        combination_list.append((i, j))

            for_loop_result = None
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
            n_r = for_loop_result[-1][0]

            # pull arms according to list n_r
            for i in range(self.d):
                for _ in range(int(n_r[i])):
                    observation = self.sample_edge(self.cost_matrix[i])
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
            self.actions_list = self.actions_list[: int(np.ceil(self.d / (2**r)))]

        if len(self.actions_list) > 1:
            print("We have more than one action in the final mathcal_A.")
            exit()
        print("sum of num_pulls", np.sum(self.num_pulls))
        return np.int_(self.actions_list[0]), np.int_(self.optimal_pi)
