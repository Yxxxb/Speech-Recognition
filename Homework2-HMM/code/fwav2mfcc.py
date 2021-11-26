# -*- coding:utf-8 -*-
"""
Author :Yxxxb & Xubing Ye
Number :1953348
Date   :2021/11/20
File   :fwav2mfcc.py
"""
from python_speech_features import *
import numpy as np
import librosa
import os


# My own mfcc method was put in self_mfcc.py, you could call it in main method.
# This function calls the python_speech_features library for comparison with the libsora library functions.
def fwav2mfcc_psf(inFilename, outFilename, folderPath, pre_emp):
    wave_data, sr = librosa.load(inFilename, sr=8000, offset=0.0)
    wavelen = len(wave_data)
    speech = np.zeros([wavelen])
    speech[0] = wave_data[0]
    speech[1:] = wave_data[1:] - pre_emp * wave_data[0:wavelen - 1]

    mfcc_feature = mfcc(speech, sr, winlen=0.025, winstep=0.01, nfilt=13, preemph=0, nfft=1024)
    d_mfcc_feat = delta(mfcc_feature, 1)
    d_mfcc_feat2 = delta(mfcc_feature, 2)
    feature1 = np.hstack((mfcc_feature, d_mfcc_feat, d_mfcc_feat2))
    feature = np.transpose(feature1)

    folder = os.path.exists(folderPath)
    if not folder:
        os.makedirs(folderPath)
    np.savetxt(outFilename, feature)


# This function calls the libsora library to select the MFCC method for this project.
# If you want to switch to a different MFCC method, you can change it in the source file.
def fwav2mfcc(wav_file):
    # Reading audio files
    y, sr = librosa.load(wav_file, 8000)

    # feature extract extraction
    fea = librosa.feature.mfcc(y, sr, n_mfcc=12, n_mels=24, n_fft=256, win_length=256, hop_length=80, lifter=12)
    stft_coff = abs(librosa.stft(y, 256, 80, 256))
    energy = np.log10(np.sum(np.square(stft_coff), 0))

    # Matrix Mosaic and regularization
    fea = np.vstack((fea, energy))
    mean = np.mean(fea, axis=1, keepdims=True)
    std = np.std(fea, axis=1, keepdims=True)
    fea = (fea - mean) / std

    # First order difference and second order difference
    delta = librosa.feature.delta(fea)
    delta_delta = librosa.feature.delta(delta)

    # concatenate
    fea = np.concatenate([fea.T, delta.T, delta_delta.T], axis=1).T
    return fea
