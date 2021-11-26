# -*- coding:utf-8 -*-
"""
Author :Yxxxb & Xubing Ye
Number :1953348
Date   :2021/11/22
File   :HMM_testing.py
"""
import numpy as np
from viterbi_dist_FR import viterbi_dist_FR


def HMM_testing(hmm, testing_list):
    num_of_error = 0
    num_of_testing = 0
    for m in range(11):
        feas = testing_list[m]
        num_of_testing = num_of_testing + len(feas)
    print('Number of testing data : ', num_of_testing)
    print('Calculating ans... (You may wait for nearly 3 minutes...)')

    for m in range(11):
        feas = testing_list[m]
        num_of_uter = len(feas)
        for u in range(0, num_of_uter):
            fea = feas[u]
            fopt_max = np.NINF
            digit = -1
            for p in range(0, 11):
                fopt = viterbi_dist_FR(hmm.HMM_mean[:, :, p], hmm.HMM_var[:, :, p], hmm.HMM_Aij[:, :, p], fea)
                if fopt > fopt_max:
                    digit = p
                    fopt_max = fopt

            if digit != m:
                num_of_error = num_of_error + 1

    print('Number of error prediction : ', num_of_error)
    accuracy = (num_of_testing - num_of_error) * 100 / num_of_testing
    return accuracy