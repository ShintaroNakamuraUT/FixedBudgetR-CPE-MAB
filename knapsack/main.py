import argparse
from collections import defaultdict

import numpy as np

from Algorithm.CSA import play_CSA
from Algorithm.MinMax_CombSA import play_MinMax_CombSA
from Algorithm.MinMax_CombSA_new import play_MinMax_CombSA_new
from Algorithm.Naive import play_Naive
from Algorithm.play_PEACE import play_PEACE
from environment import actions_list_maker


def get_args():
    parser = argparse.ArgumentParser(description="argparse script")
    parser.add_argument(
        "-num_exp",
        "--num_exp",
        type=int,
        default=50,
        help="The number of times we run the experiment",
    )
    parser.add_argument(
        "-total_weight",
        "--total_weight",
        type=int,
        default=200,
        help="Total weight",
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

    for budget in [10000]:
        RESULT = {}
        # N is the number of items
        for N in range(10, 101, 10):
            result_list = []
            (
                correct_cnt_PEACE,
                correct_cnt_CSA,
                correct_cnt_Naive,
                correct_cnt_MinMax_CombSA_non_linear_solver,
                correct_cnt_MinMax_CombSA_Lagrange,
                correct_cnt_MinMax_CombSA_new,
            ) = (0, 0, 0, 0, defaultdict(int), defaultdict(int))

            for num_exp in range(args.num_exp):
                print("N = {}, num_exp = {}".format(N, num_exp + 1))

                total_weight = args.total_weight

                R_sub_Gaussian = 1

                # Uniformly sample the weights from {1, ..., 200}.
                weights_lower, weights_upper, value_param = 1, 200, 0.1
                weights = np.random.randint(weights_lower, weights_upper, N)

                # Generate the value for each item.
                value_width = 4
                values = weights * (1 + value_param * np.random.rand(N))

                if args.actionset_size == "poly":
                    K_actions = 1000
                    # make a list of actions
                    actions_list = None
                    dfs = actions_list_maker(
                        weights,
                        total_weight,
                        value_width,
                        K_actions,
                        value_param,
                    )
                    dfs.solver()
                    actions_list = dfs.actions_list

                    # Play CSA algorithm
                    print("CSA algorithm")
                    result_CSA = play_CSA(
                        args.actionset_size,
                        actions_list,
                        total_weight,
                        values,
                        weights,
                        budget,
                        R_sub_Gaussian,
                    ).play()
                    if np.all(result_CSA[0] == result_CSA[1]):
                        correct_cnt_CSA += 1

                    # Play MinMax_CombSA algorithm
                    # for BETA in [0.2, 0.4, 0.6, 0.8]:
                    #     print(
                    #         "MinMax_CombSA algorithm (Lagrange) with BETA = {}".format(
                    #             BETA
                    #         )
                    #     )
                    #     result_MinMax_CombSA = play_MinMax_CombSA(
                    #         actions_list=actions_list,
                    #         values=values,
                    #         budget=budget,
                    #         R_sub_Gaussian=R_sub_Gaussian,
                    #         allocatin_vector_type="Lagrange",
                    #         beta=BETA,
                    #     ).play()
                    #     if np.all(result_MinMax_CombSA[0] == result_MinMax_CombSA[1]):
                    #         correct_cnt_MinMax_CombSA_Lagrange[BETA] += 1
                    for BETA in [0.2, 0.4, 0.6, 0.8]:
                        print(
                            "New MinMax_CombSA algorithm (Lagrange) with BETA = {}".format(
                                BETA
                            )
                        )
                        result_MinMax_CombSA_new = play_MinMax_CombSA_new(
                            actions_list=actions_list,
                            values=values,
                            budget=budget,
                            R_sub_Gaussian=R_sub_Gaussian,
                            allocatin_vector_type="Lagrange",
                            beta=BETA,
                        ).play()
                        if np.all(
                            result_MinMax_CombSA_new[0] == result_MinMax_CombSA_new[1]
                        ):
                            correct_cnt_MinMax_CombSA_new[BETA] += 1

                    # Play the PEACE algorithm
                    print("PEACE algorithm")
                    result_PEACE = play_PEACE(
                        actions_list=np.array(actions_list),
                        values=values,
                        budget=budget,
                        R_sub_Gaussian=R_sub_Gaussian,
                    ).play()
                    if np.all(result_PEACE[0] == result_PEACE[1]):
                        correct_cnt_PEACE += 1

                    print(weights)
                    print(values)
                    print("Number of correct answer")
                    print(
                        "PEACE: {}, CSA: {}, MinMax_CombSA (non_linear_solver):{}, MinMax_CombSA (Lagrange):{}, New Minmax_Comb:{}".format(
                            correct_cnt_PEACE,
                            correct_cnt_CSA,
                            correct_cnt_MinMax_CombSA_non_linear_solver,
                            correct_cnt_MinMax_CombSA_Lagrange,
                            correct_cnt_MinMax_CombSA_new,
                        )
                    )
                elif args.actionset_size == "exp":
                    # Play CSA algorithm
                    print("CSA algorithm")
                    result_CSA = play_CSA(
                        args.actionset_size,
                        None,
                        total_weight,
                        values,
                        weights,
                        budget,
                        R_sub_Gaussian,
                    ).play()
                    if np.all(result_CSA[0] == result_CSA[1]):
                        correct_cnt_CSA += 1
                    # Play Naive algorithm
                    result_Naive = play_Naive(
                        args.actionset_size,
                        None,
                        total_weight,
                        values,
                        weights,
                        budget,
                        R_sub_Gaussian,
                    ).play()
                    if np.all(result_Naive[0] == result_Naive[1]):
                        correct_cnt_Naive += 1
                    print(N, correct_cnt_CSA, correct_cnt_Naive)
            print(
                "The result of N={}. PEACE:{}, CSA: {}, Naive:{} MinMax_CombSA (non_linear_solver):{}, MinMax_CombSA (Lagrange):{}, New MinMax_CombSA:{}".format(
                    N,
                    correct_cnt_PEACE,
                    correct_cnt_CSA,
                    correct_cnt_Naive,
                    correct_cnt_MinMax_CombSA_non_linear_solver,
                    correct_cnt_MinMax_CombSA_Lagrange,
                    correct_cnt_MinMax_CombSA_new,
                )
            )
            RESULT[N] = (
                correct_cnt_PEACE,
                correct_cnt_CSA,
                correct_cnt_Naive,
                correct_cnt_MinMax_CombSA_non_linear_solver,
                correct_cnt_MinMax_CombSA_Lagrange,
                correct_cnt_MinMax_CombSA_new,
            )

            import pickle

            print(RESULT)
            if args.actionset_size == "exp":
                with open(
                    "result/result_CSA/result_CSA_{}_{}.txt".format(
                        args.actionset_size, budget
                    ),
                    "wb",
                ) as f:
                    pickle.dump(RESULT, f)
            if args.actionset_size == "poly":
                with open(
                    "result/result_poly/result_{}_{}_{}_{}_{}_{}_{}_{}.txt".format(
                        args.actionset_size,
                        budget,
                        K_actions,
                        weights_lower,
                        weights_upper,
                        value_param,
                        args.total_weight,
                        args.num_exp,
                    ),
                    "wb",
                ) as f:
                    pickle.dump(RESULT, f)
