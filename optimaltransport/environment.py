import numpy as np

from compute_OT import Compute_OT


class actions_list_maker:
    def __init__(self, supply_vector, demand_vector, K_actions):
        self.supply_vector = supply_vector
        self.demand_vector = demand_vector
        self.K_actions = K_actions
        self.actions_list = []
        self.d = len(self.supply_vector)

    def solver(self):
        for _ in range(1, self.K_actions + 1):
            dummy_cost_matrix = np.random.rand(self.d, self.d)
            solution, _ = Compute_OT(
                self.supply_vector, self.demand_vector, dummy_cost_matrix
            ).compute_OT()
            solution_flatten = solution.flatten().tolist()

            if solution_flatten not in self.actions_list:
                self.actions_list.append(solution_flatten)
