# -*- coding:utf-8 -*-
"""
Author :Yxxxb & Xubing Ye
Number :1953348
Date   :2021/11/22
File   :tool_function.py
"""
import numpy as np


# tool function file for HMM_training / HMM_testing / EN_HMM_FR / Viterbi

def tempMat(M, temp):
    temp1 = np.append(temp, M, axis=1)
    temp2 = np.append(temp1, temp, axis=1)
    return temp2


def getThirdDim(vec, k):
    newVec = np.zeros((vec.shape[0], vec.shape[1]))
    for i, value1 in enumerate(vec):
        for j, value2 in enumerate(value1):
            newVec[i][j] = value2[k]
    return newVec


def logGaussian(mean_i, var_i, o_i):
    dim = len(var_i)
    log_b = -1 / 2 * (dim * np.log(2 * np.pi) + np.sum(np.log(var_i)) + np.sum((o_i - mean_i) * (o_i - mean_i) / var_i))
    return log_b


def log_sum_alpha(log_alpha_t, aij_j):
    len_x = len(log_alpha_t)
    y = np.full((len_x,), np.NINF)
    ymax = np.NINF
    for i in range(len_x):
        if aij_j[i] > 0:
            y[i] = log_alpha_t[i] + np.log(aij_j[i])
        if y[i] > ymax:
            ymax = y[i]
    if ymax == np.inf:
        return ymax
    else:
        sum_exp = 0
        for i in range(len_x):
            if ymax == np.NINF and y[i] == np.NINF:
                sum_exp = sum_exp + 1
            else:
                sum_exp = sum_exp + np.exp(y[i] - ymax)
        return ymax + np.log(sum_exp)


def log_sum_beta(aij_i, mean, var, obs, beta_t1):
    len_x = mean.shape[1]
    y = np.full((len_x,), np.NINF)
    ymax = np.NINF
    for j in range(len_x):
        y[j] = np.log(aij_i[j]) + logGaussian(mean[:, j], var[:, j], obs) + beta_t1[j]
        if y[j] > ymax:
            ymax = y[j]
    if ymax == np.inf:
        return ymax
    else:
        sum_exp = 0
        for i in range(len_x):
            if ymax == np.NINF and y[i] == np.NINF:
                sum_exp = sum_exp + 1
            else:
                sum_exp = sum_exp + np.exp(y[i] - ymax)
        return ymax + np.log(sum_exp)
