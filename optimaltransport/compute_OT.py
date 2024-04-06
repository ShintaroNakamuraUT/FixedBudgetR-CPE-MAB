import numpy as np
from pulp import PULP_CBC_CMD, LpMinimize, LpProblem, LpVariable, lpSum, value


class Compute_OT:
    """Reference: https://local-optima.hateblo.jp/entry/pulp2-transportation"""

    def __init__(self, suppliers, demanders, cost) -> None:
        self.supplies = suppliers
        self.demands = demanders
        self.cost = cost  # 二次元は配列で受け取る

    def compute_OT(
        self,
    ):
        cost_dict = {}
        row, column = len(self.cost), len(self.cost[0])
        for i in range(row):
            for j in range(column):
                cost_dict[(i, j)] = self.cost[i][j]
        model = LpProblem("Transportation", LpMinimize)
        x = {}
        for i, j in cost_dict:
            x[i, j] = LpVariable(
                "x{}-{}".format(i, j), lowBound=0, upBound=self.demands[j]
            )
        # 目的関数
        model += lpSum([cost_dict[i, j] * x[i, j] for i, j in cost_dict]), "Objective"
        # 制約条件
        # 工場の出荷上限
        for i, Ci in enumerate(self.supplies):
            model += lpSum(
                [x[i, j] for j in range(len(self.demands))]
            ) <= Ci, "Capacity{}".format(i)

        # 顧客の需要
        for j, dj in enumerate(self.demands):
            model += lpSum(
                [x[i, j] for i in range(len(self.supplies))]
            ) == dj, "demand{}".format(j)
        # 求解
        model.solve(PULP_CBC_CMD(msg=0))

        # 結果の確認
        output = [
            [0 for _ in range(len(self.demands))] for i in range(len(self.supplies))
        ]

        x_values = list(x.values())
        supplies_num, demands_num = len(self.supplies), len(self.demands)
        for row in range(supplies_num):
            for column in range(demands_num):
                if x_values[row * demands_num + column].varValue > 1e-4:
                    output[row][column] = x_values[row * demands_num + column].varValue
        total_cost = value(model.objective)
        return np.array(output), total_cost
