# -*- coding:utf-8 -*-
"""
Author :Yxxxb & Xubing Ye
Number :1953348
Date   :2021/11/21
File   :viterbi_dist_FR.py
"""
import numpy as np
from tool_function import logGaussian, tempMat


def viterbi_dist_FR(mean, var, aij, obs):
    # initial and preprocess the initial state and end state of mean and VAR.
    dim, t_len = obs.shape
    temp = np.zeros((dim, 1))
    temp[:, 0] = np.NAN

    mean = tempMat(mean, temp)
    var = tempMat(var, temp)
    aij[aij.shape[0] - 1][aij.shape[1] - 1] = 1

    m_len = mean.shape[1]
    fjt = np.zeros((m_len, t_len))

    # Initialize the correlation function for the forward and backward parameters.
    def tempForward(M):
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                M[i][j] = np.NINF
        return M

    fjt = tempForward(fjt)
    s_chain = np.empty((m_len, t_len), dtype=object)

    for j in range(1, m_len - 1):
        if aij[0, j] > 0:
            fjt[j][0] = np.log(aij[0, j]) + logGaussian(mean[:, j], var[:, j], obs[:, 0])
        else:
            fjt[j][0] = np.NINF
        if fjt[j][0] > np.NINF:
            s_chain[j][0] = np.array([0, j])

    for t in range(1, t_len):
        for j in range(1, m_len - 1):
            f_max = np.NINF
            i_max = -1
            f = np.NINF
            for i in range(1, j + 1):
                if fjt[i, t - 1] > np.NINF:
                    if aij[i, j] > 0:
                        f = fjt[i, t - 1] + np.log(aij[i, j]) + logGaussian(mean[:, j], var[:, j], obs[:, t])
                    else:
                        f = np.NINF
                if f > f_max:
                    f_max = f
                    i_max = i
            if i_max != -1:
                s_chain[j, t] = np.array([s_chain[i_max, t - 1], j], dtype=object)
                fjt[j, t] = f_max

    fopt = np.NINF
    for i in range(1, m_len - 1):
        if aij[i, m_len - 1] > 0:
            f = fjt[i, t_len - 1] + np.log(aij[i, m_len - 1])
        else:
            f = np.NINF
        if f > fopt:
            fopt = f

    return fopt
