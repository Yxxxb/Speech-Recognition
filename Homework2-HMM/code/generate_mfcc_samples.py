# -*- coding:utf-8 -*-
"""
Author :Yxxxb & Xubing Ye
Number :1953348
Date   :2021/11/20
File   :generate_mfcc_samples.py
"""
import os
import numpy as np
from fwav2mfcc import fwav2mfcc
from self_mfcc import self_mfcc

outfile_format = 'htk'
frame_size_sec = 0.025
frame_shift_sec = 0.010
use_hamming = 1
pre_emp = 0
bank_no = 26
cep_order = 12
lifter = 22
delta_win = 2
delta_win_weight = np.ones((1, 2 * delta_win + 1))


# For your convenience in running the project, the MFCC matrix is not saved locally.
# You can call this function if you want to save and access the MFCC matrix locally.
def generate_mfcc_samples():
    print('generate mfcc samples successfully!')
    filePath = "./wav"
    fileList = os.listdir(filePath)

    for path in fileList:
        wavPath = filePath + '/' + path
        wavList = os.listdir(wavPath)
        for wav in wavList:
            inFilename = wavPath + '/' + wav
            outFilename = './mfcc' + '/' + path + '/' + wav + '.txt'
            folderPath = './mfcc' + '/' + path

            # fwav2mfcc(inFilename, outFilename, folderPath, pre_emp)
