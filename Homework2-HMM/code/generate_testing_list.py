# -*- coding:utf-8 -*-
"""
Author :Yxxxb & Xubing Ye
Number :1953348
Date   :2021/11/21
File   :generate_testing_list.py
"""
import os
from fwav2mfcc import fwav2mfcc


def generate_testing_list():
    print('generate testing list successfully!')
    key = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'O', 'Z']
    testing_folder_list = ['AH', 'AR', 'AT', 'BC', 'BE', 'BM', 'BN', 'CC', 'CE', 'CP', 'DF', 'DJ', 'ED', 'EF', 'ET',
                           'FA', 'FG', 'FH',
                           'FM', 'FP', 'FR', 'FS', 'FT', 'GA', 'GP', 'GS', 'GW', 'HC', 'HJ', 'HM', 'HR', 'IA', 'IB',
                           'IM', 'IP', 'JA']
    test_feature_list = []
    for item in key:
        feature_list = []
        for dir in os.listdir("./wav"):
            if dir in testing_folder_list:
                filePath = "./wav" + '/' + dir
                txtList = os.listdir(filePath)
                for txt in txtList:
                    if item in txt:
                        fea = fwav2mfcc(filePath + '/' + txt)
                        feature_list.append(fea)
        test_feature_list.append(feature_list)

    return test_feature_list
