# -*- coding:utf-8 -*-
"""
Author :Yxxxb & Xubing Ye
Number :1953348
Date   :2021/11/23
File   :self_mfcc.py
"""

import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct


def self_mfcc(inFileName):
    # import and show signal
    SamplingRate, signal = scipy.io.wavfile.read(inFileName)

    # Pre-Emphasis
    alpha = 0.98
    signal = signal.copy()
    for i in range(1, len(signal), 1):
        signal[i] = signal[i] - alpha * signal[i - 1]
    LengthofSignal = len(signal)

    # Framing
    SizeofFrame = 0.025  # Set the frame length and step length.
    StrideofFrame = 0.01
    LengthofFrame = int(SizeofFrame * SamplingRate)
    step = int(StrideofFrame * SamplingRate)
    num_frames = int(np.ceil(float(LengthofSignal - LengthofFrame) / step))
    LengthofPad = num_frames * step + LengthofFrame

    z = np.zeros((LengthofPad - LengthofSignal))
    pad = np.append(signal, z)
    # Array construction and combination this line of index processing code from the reference
    index = np.tile(np.arange(0, LengthofFrame), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * step, step), (LengthofFrame, 1)).T
    framed = pad[index.astype(np.int32, copy=False)]

    # Windowing
    # Hamming Window
    nn = [i for i in range(LengthofFrame)]
    framed *= 0.54 - 0.46 * np.cos(np.multiply(nn, 2 * np.pi) / (LengthofFrame - 1))

    # Fourier-Transform and Power Spectrum
    NFFT = 512

    # from myself (Sin Wave)
    fft = np.sin(2 * np.pi * 3 * (framed / 512))  # Sin wave method
    # from fft package
    fft1 = np.absolute(np.fft.rfft(framed, NFFT))  # FFT
    fft2 = ((1.0 / NFFT) * ((fft1) ** 2))

    # MMFC & mel
    numofFilter = 40
    LowerBound = 0
    HigherBound = (2595 * np.log10(1 + (SamplingRate / 2) / 700))
    mel = np.linspace(LowerBound, HigherBound, numofFilter + 1)
    hz = (700 * (10 ** (mel / 2595) - 1))
    bin = np.floor((NFFT + 1) * hz / SamplingRate)
    fbank = np.zeros((numofFilter, int(np.floor(NFFT / 2 + 1))))

    for i in range(1, numofFilter):
        for j in range(bin[i - 1], bin[i]):
            fbank[i - 1, j] = (j - bin[i - 1]) / (bin[i] - bin[i - 1])
        for j in range(bin[i], bin[i + 1]):
            fbank[i - 1, j] = (bin[i + 1] - j) / (bin[i + 1] - bin[i])

    ans = np.dot(fft2, fbank.T)
    ans = np.where(ans == 0, np.finfo(float).eps, ans)
    # take logarithm
    ans = np.log10(ans)

    # dct
    num_ceps = 12
    dct_ans = dct(ans, type=2, axis=-1, norm='ortho')[1: (num_ceps + 1)]

    return dct_ans
