import argparse
import pickle
from collections import defaultdict

import numpy as np

from Algorithm.CSA import play_CSA
from Algorithm.MinMax_CombSA_new import play_MinMax_CombSA_new
from Algorithm.Peace import play_Peace
from environment import actions_list_maker


def get_args():
    parser = argparse.ArgumentParser(description="argparse script")
    parser.add_argument(
        "-num_exp",
        "--num_exp",
        type=int,
        default=100,
        help="The number of times we run the experiment",
    )
    parser.add_argument(
        "-budget",
        "--budget",
        type=int,
        default=5000,
        help="Budget",
    )
    parser.add_argument(
        "-actionset_size",
        "--actionset_size",
        choices=["exp", "poly"],
        default="poly",
        help="The assumption of the size of the action set.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    for budget in [args.budget]:
        RESULT = {}
        for m in range(3, 12, 1):
            result_list = []
            correct_cnt_PEACE, correct_cnt_CSA, correct_cnt_MinMax_CombSA_new = (
                0,
                0,
                defaultdict(int),
            )
            for num_exp in range(args.num_exp):
                print("m = {}, num_exp = {}".format(m, num_exp + 1))
                R_sub_Gaussian = 1

                # supply vector and demand vector
                supply_vector = [(i + 1) for i in range(m)]
                demand_vector = [(i + 1) for i in range(m)]

                # Generate cost matrix
                cost_matrix = np.random.rand(m, m)

                if args.actionset_size == "poly":
                    K_actions = 100
                    # Make a list of actions
                    actions_list = None

                    OT_solver = actions_list_maker(
                        supply_vector,
                        demand_vector,
                        K_actions,
                    )
                    OT_solver.solver()
                    actions_list = OT_solver.actions_list
                    print("The size of the action set is ", len(actions_list))
                    # Play CSA algorithm
                    print("CSA algorithm")
                    result_CSA = play_CSA(
                        args.actionset_size,
                        actions_list,
                        supply_vector,
                        demand_vector,
                        cost_matrix,  # The true cost matrix
                        budget,
                        R_sub_Gaussian,
                    ).play()
                    if np.all(result_CSA[0] == result_CSA[1]):
                        correct_cnt_CSA += 1

                    # Play MinMax_CombSA algorithm
                    for BETA in [0.2, 0.4]:
                        print(
                            "New MinMax_CombSA algorithm (Lagrange) with BETA = {}".format(
                                BETA
                            )
                        )
                        result_MinMax_CombSA_new = play_MinMax_CombSA_new(
                            actions_list,
                            supply_vector,
                            demand_vector,
                            cost_matrix,  # The true cost matrix
                            budget,
                            R_sub_Gaussian,
                            allocation_vector_type="Lagrange",
                            beta=BETA,
                        ).play()
                        if np.all(
                            result_MinMax_CombSA_new[0] == result_MinMax_CombSA_new[1]
                        ):
                            correct_cnt_MinMax_CombSA_new[BETA] += 1
                    # Play the PEACE algorithm
                    print("PEACE algorithm")
                    result_PEACE = play_Peace(
                        np.array(actions_list),
                        supply_vector,
                        demand_vector,
                        cost_matrix,
                        budget,
                        R_sub_Gaussian,
                    ).play()
                    if np.all(result_PEACE[0] == result_PEACE[1]):
                        correct_cnt_PEACE += 1
                print(
                    "The result of m={}. \n PEACE:{}, CSA: {}, New MinMax_CombSA:{}".format(
                        m,
                        correct_cnt_PEACE,
                        correct_cnt_CSA,
                        correct_cnt_MinMax_CombSA_new,
                    )
                )
            RESULT[m] = (
                correct_cnt_PEACE,
                correct_cnt_CSA,
                correct_cnt_MinMax_CombSA_new,
            )
            print(RESULT)
            if args.actionset_size == "poly":
                with open(
                    "result/result_{}_{}.txt".format(args.budget, args.num_exp),
                    "wb",
                ) as f:
                    pickle.dump(RESULT, f)
