# -*- coding:utf-8 -*-
"""
Author :Yxxxb & Xubing Ye
Number :1953348
Date   :2021/11/21
File   :EM_HMM_FR.py
"""

from tool_function import *


def EM_HMM_FR(mean, var, aij, obs):
    # initialize
    dim = 39
    T = obs.shape[1]
    addDim = np.full((dim, 1), np.nan)
    mean = np.hstack((addDim, mean, addDim))
    var = np.hstack((addDim, var, addDim))
    aij[-1][-1] = 1
    N = mean.shape[1]
    log_alpha = np.full((N, T + 1), np.NINF)
    log_beta = np.full((N, T + 1), np.NINF)

    # calculate alpha
    for i in range(N):
        if aij[0, i] > 0:
            log_alpha[i, 0] = np.log(aij[0, i]) + logGaussian(mean[:, i], var[:, i], obs[:, 1])

    for t in range(1, T):
        for j in range(1, N - 1):
            log_alpha[j, t] = log_sum_alpha(log_alpha[1:N - 1, t - 1], aij[1:N - 1, j]) + logGaussian(mean[:, j],
                                                                                                      var[:, j],
                                                                                                      obs[:, t])
    log_alpha[N - 1, T] = log_sum_alpha(log_alpha[1:N - 1, T - 1], aij[1:N - 1, N - 1])

    # calculate beta
    for t in range(N):
        if aij[t, N - 1] > 0:
            log_beta[t, T - 1] = np.log(aij[t, N - 1])
    for t in range(T - 2, 0, -1):
        for i in range(1, N - 1):
            log_beta[i, t] = log_sum_beta(aij[i, 1:N - 1], mean[:, 1: N - 1], var[:, 1: N - 1], obs[:, t + 1],
                                          log_beta[
                                          1: N - 1,
                                          t + 1])
    log_beta[N - 1, 0] = log_sum_beta(aij[0, 1:N - 1], mean[:, 1:N - 1], var[:, 1:N - 1], obs[:, 0],
                                      log_beta[1:N - 1, 0])

    # calculate Xi
    log_Xi = np.full((N, N, T), np.NINF)
    for t in range(T - 1):
        for j in range(1, N - 1):
            for i in range(1, N - 1):
                if aij[i, j] > 0:
                    log_Xi[i, j, t] = log_alpha[i, t] + np.log(aij[i, j]) + logGaussian(
                        mean[:, j], var[:, j], obs[:, t + 1]) + log_beta[j, t + 1] - log_alpha[N - 1, T]
    for i in range(N):
        log_Xi[i][N - 1][T - 1] = log_alpha[i][T - 1] - np.inf - log_alpha[N - 1][T]

    # calculate gamma
    log_gamma = np.full((N, T), np.NINF)
    for t in range(T):
        for i in range(1, N - 1):
            log_gamma[i][t] = log_alpha[i][t] + log_beta[i][t] - log_alpha[N - 1][T]
    gamma = np.exp(log_gamma)

    # calculate return  : mean_numerator, var_numerator, aij_numerator, denominator, log_likelihood, likelihood
    mean_numerator = np.zeros((dim, N))
    var_numerator = np.zeros((dim, N))
    denominator = np.zeros((N,))
    aij_numerator = np.zeros((N, N))
    for j in range(1, N - 1):
        for t in range(T):
            for i in range(mean_numerator.shape[0]):
                mean_numerator[i][j] = mean_numerator[i][j] + gamma[j][t] * obs[i][t]
                var_numerator[i][j] = var_numerator[i][j] + gamma[j][t] * obs[i][t] * obs[i][t]
            denominator[j] = denominator[j] + gamma[j][t]

    for i in range(1, N - 1):
        for j in range(1, N - 1):
            for t in range(T):
                aij_numerator[i][j] = aij_numerator[i][j] + np.exp(log_Xi[i][j][t])

    log_likelihood = log_alpha[N - 1][T]
    likelihood = np.exp(log_alpha[N - 1][T])

    return mean_numerator, var_numerator, aij_numerator, denominator, log_likelihood, likelihood
