# -*- coding:utf-8 -*-
"""
Author :Yxxxb & Xubing Ye
Number :1953348
Date   :2021/11/21
File   :generate_training_list.py
"""
import os
from fwav2mfcc import fwav2mfcc


def generate_training_list():
    print('generate training list successfully!')
    key = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'O', 'Z']
    training_folder_list = ['AE', 'AJ', 'AL', 'AW', 'BD', 'CB', 'CF', 'CR', 'DL', 'DN', 'EH', 'EL', 'FC', 'FD', 'FF',
                            'FI', 'FJ', 'FK',
                            'FL', 'GG']
    train_feature_list = []
    for item in key:
        feature_list = []
        for dir in os.listdir("./wav"):
            if dir in training_folder_list:
                filePath = "./wav" + '/' + dir
                txtList = os.listdir(filePath)
                for txt in txtList:
                    if item in txt:
                        fea = fwav2mfcc(filePath + '/' + txt)
                        feature_list.append(fea)
        train_feature_list.append(feature_list)

    return train_feature_list
