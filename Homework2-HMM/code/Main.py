# -*- coding:utf-8 -*-
"""
Author :Yxxxb & Xubing Ye
Number :1953348
Date   :2021/11/20
File   :Main.py
"""

import warnings
from generate_mfcc_samples import generate_mfcc_samples
from generate_training_list import generate_training_list
from generate_testing_list import generate_testing_list
from HMM_testing import HMM_testing
from HMM_training import HMM

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # generate_mfcc_samples()
    training_name_list = generate_training_list()
    testing_name_list = generate_testing_list()

    DIM = 39
    num_of_model = 11
    num_of_state_start = 12
    num_of_state_end = 15

    for num_of_state in range(num_of_state_start, num_of_state_end + 1):
        print('num of state  : ', num_of_state)
        model = HMM(training_name_list, DIM, num_of_model, num_of_state)
        model.HMM_training()
        accuracy_rate = HMM_testing(model, testing_name_list)
        print('accuracy rate : ', accuracy_rate)
