# -*- coding:utf-8 -*-
"""
Author :Yxxxb & Xubing Ye
Number :1953348
Date   :2021/11/21
File   :HMM_training.py
"""
import numpy as np
from EM_HMM_FR import EM_HMM_FR


class HMM:
    def __init__(self, training_name_list, DIM, num_of_model, num_of_state):
        # initialize model nature
        self.DIM = DIM
        self.training_name_list = training_name_list
        self.num_of_model = num_of_model
        self.num_of_state = num_of_state
        self.HMM_mean = np.zeros([DIM, num_of_state, num_of_model])
        self.HMM_var = np.zeros([DIM, num_of_state, num_of_model])
        self.HMM_Aij = np.zeros([num_of_state + 2, num_of_state + 2, num_of_model])

    def initialize_model(self):
        # initialize model
        print('initialize model successfully!')
        sum_of_features = np.zeros((self.DIM,))
        sum_of_features_square = np.zeros((self.DIM,))
        num_of_feature = 0

        for m in range(self.num_of_model):
            feature_list = self.training_name_list[m]
            num_of_uter = len(feature_list)
            for u in range(0, num_of_uter):
                feature = feature_list[u]
                num_of_feature = num_of_feature + feature.shape[1]
                sum_of_features = sum_of_features + np.sum(feature, axis=1)
                sum_of_features_square = sum_of_features_square + np.sum(pow(feature, 2), axis=1)

        self.calculate_initial_EM_HMM_items(sum_of_features, sum_of_features_square, num_of_feature)

    def calculate_initial_EM_HMM_items(self, sum_of_features, sum_of_features_square, num_of_feature):
        # initialize aij / mean / var
        for k in range(self.num_of_model):
            for m in range(self.num_of_state):
                for idx in range(39):
                    self.HMM_mean[idx, m, k] = sum_of_features[idx] / num_of_feature
                    self.HMM_var[idx, m, k] = sum_of_features_square[idx] / num_of_feature - self.HMM_mean[idx, m, k] * \
                                              self.HMM_mean[idx, m, k]
            for i in range(1, self.num_of_state + 1):
                self.HMM_Aij[i, i + 1, k] = 0.4
                self.HMM_Aij[i, i, k] = 1 - self.HMM_Aij[i, i + 1, k]

            self.HMM_Aij[0, 1, k] = 1.0

    def HMM_training(self):
        # initialize model
        self.initialize_model()
        num_of_iteration = 2
        log_likelihood_iter = np.zeros((num_of_iteration, 1))
        likelihood_iter = np.zeros((num_of_iteration, 1))

        # training
        for i in range(num_of_iteration):
            print('------------------------------------------')
            print('start training   epoch :', i + 1)

            sum_mean_numerator = np.zeros((self.DIM, self.num_of_state, self.num_of_model))
            sum_var_numerator = np.zeros((self.DIM, self.num_of_state, self.num_of_model))
            sum_aij_numerator = np.zeros((self.num_of_state, self.num_of_state, self.num_of_model))
            sum_denominator = np.zeros((self.num_of_state, self.num_of_model))
            log_likelihood = 0
            likelihood = 0

            for m in range(self.num_of_model):
                print('Number of training class : ', m + 1)
                feature_list = self.training_name_list[m]
                num_of_uter = len(feature_list)
                for u in range(0, num_of_uter):
                    feature = feature_list[u]
                    mean_numerator, var_numerator, aij_numerator, denominator, log_likelihood_i, likelihood_i = EM_HMM_FR(
                        self.HMM_mean[:, :, m], self.HMM_var[:, :, m], self.HMM_Aij[:, :, m], feature)

                    sum_mean_numerator[:, :, m] = sum_mean_numerator[:, :, m] + mean_numerator[:, 1: -1]
                    sum_var_numerator[:, :, m] = sum_var_numerator[:, :, m] + var_numerator[:, 1: -1]
                    sum_aij_numerator[:, :, m] = sum_aij_numerator[:, :, m] + aij_numerator[1: - 1, 1: -1]
                    sum_denominator[:, m] = sum_denominator[:, m] + denominator[1: -1]

                    log_likelihood = log_likelihood + log_likelihood_i
                    likelihood = likelihood + likelihood_i

            for k in range(self.num_of_model):
                for n in range(self.num_of_state):
                    self.HMM_mean[:, n, k] = sum_mean_numerator[:, n, k] / sum_denominator[n, k]
                    self.HMM_var[:, n, k] = sum_var_numerator[:, n, k] / sum_denominator[n, k] - self.HMM_mean[:, n,
                                                                                                 k] * self.HMM_mean[
                                                                                                      :,
                                                                                                      n, k]
            for k in range(self.num_of_model):
                for ii in range(1, self.num_of_state + 1):
                    for j in range(1, self.num_of_state + 1):
                        self.HMM_Aij[ii, j, k] = sum_aij_numerator[ii - 1, j - 1, k] / sum_denominator[ii - 1, k]
                self.HMM_Aij[self.num_of_state, self.num_of_state + 1, k] = 1 - self.HMM_Aij[
                    self.num_of_state, self.num_of_state, k]
            self.HMM_Aij[self.num_of_state + 1, self.num_of_state + 1, -1] = 1
            log_likelihood_iter[i] = log_likelihood
            likelihood_iter[i] = likelihood
