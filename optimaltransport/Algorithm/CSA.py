import numpy as np


class play_CSA:
    def __init__(
        self,
        actionset_size,
        actions_list,
        supply_vector,
        demand_vector,
        cost_matrix,
        budget,
        R_sub_Gaussian,
    ):
        self.actionset_size = actionset_size
        self.actions_list = actions_list
        self.supply_vector = supply_vector
        self.demand_vector = demand_vector
        self.num_of_supplier = len(cost_matrix)
        self.cost_matrix = cost_matrix.flatten()  # We convert cost matrix to 1D vector
        self.budget = budget
        self.R_sub_Gaussian = R_sub_Gaussian

        self.d = len(self.cost_matrix)

        self.sample_mean_values = [0 for row in range(self.d)]
        self.num_pulls = [0 for row in range(self.d)]

        if self.actionset_size == "poly":
            self.actions_list = actions_list
            self.K_actions = len(self.actions_list)
            # Find the true optimal action
            true_reward_list = [
                [np.dot(np.array(self.actions_list[i]), np.array(self.cost_matrix)), i]
                for i in range(self.K_actions)
            ]
            true_reward_list.sort()
            if self.actionset_size == "poly":
                self.optimal_pi = np.int_(self.actions_list[true_reward_list[-1][1]])

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
        return pi

    def T_tilde(self, t):
        if t == 0:
            return 0
        tilde_log = self.tilde_log(self.d)
        return int(np.ceil((self.budget - self.d) / (tilde_log * (self.d - t + 1))))

    def play(self):
        F_t, S_t = {}, []

        for t in range(1, self.d + 1):
            prev = self.T_tilde(t - 1)

            # The total number of pulls at round t is tilde_T
            tilde_T = self.T_tilde(t) - prev
            for i in range(self.d):
                if i in F_t:
                    continue
                for _ in range(tilde_T):
                    observation = self.sample_edge(self.cost_matrix[i])
                    self.sample_mean_values[i] = (
                        self.sample_mean_values[i] * self.num_pulls[i] + observation
                    ) / (self.num_pulls[i] + 1)
                    self.num_pulls[i] += 1
            pi_t = self.COracle(S_t)
            if pi_t is None:
                break

            compare_list = []
            for i in range(self.d):
                if i in F_t:
                    continue
                # Search \Tilde{\pi}^{e}
                tilde_pi_te = None

                # i corresponds to row = i//3 and column = i % 3
                row = i // self.num_of_supplier
                column = i % self.num_of_supplier

                for x_e in range(
                    min(self.supply_vector[row], self.demand_vector[column]) + 1
                ):
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

        answer_pi = np.array([-1 for _ in range(self.d)])
        for e, x_e in S_t:
            answer_pi[e] = x_e
        print("num_pulls", np.sum(self.num_pulls))
        return answer_pi, self.optimal_pi
