# -*- coding:utf-8 -*-
"""
Author :Yxxxb & Xubing Ye
Number :1953348
Date   :2021/10/17
File   :Speech Signal Processing (MFCC).py
"""

import numpy as np
import scipy.io.wavfile
import matplotlib.pylab as plt
from scipy.fftpack import dct

# import and show signal 导入
sample_rate, signal = scipy.io.wavfile.read('OSR_us_000_0010_8k.wav')  # File assumed to be in the same directory
print("sample rate : " + str(sample_rate))
signal = signal[int(5 * sample_rate):int(9 * sample_rate)]  # 截取中间段声音
c1 = dct(signal, type=2, axis=-1, norm='ortho')
plt.plot(c1)
plt.title('import and show signal', fontsize=12, color='black')
plt.savefig(f'./out/output1_import and show signal.png')
plt.show()

# Pre-Emphasis 预加重
pre_emphasis = 0.97
# 这个步骤在群里有讨论过 那么按照网上的来做的话将0位置保留 其余进行预处理
emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
c2 = dct(emphasized_signal, type=2, axis=-1, norm='ortho')
plt.plot(c2)
plt.title('Pre-Emphasis', fontsize=12, color='black')
plt.savefig(f'./out/output2_Pre-Emphasis.png')
plt.show()

# Framing
frame_size = 0.025  # 设置帧长以及步幅
frame_stride = 0.01
frame_length, frame_step = int(round(frame_size * sample_rate)), int(round(frame_stride * sample_rate))
signal_length = len(emphasized_signal)
num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

pad_signal_length = num_frames * frame_step + frame_length
z = np.zeros((pad_signal_length - signal_length))
pad_signal = np.append(emphasized_signal, z)
indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
    np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
frames = pad_signal[indices.astype(np.int32, copy=False)]

c3 = dct(frames, type=2, axis=-1, norm='ortho')
plt.plot(c3)
plt.title('Frames', fontsize=12, color='black')
plt.savefig(f'./out/output3_Frames.png')
plt.show()

# Windowing 加窗
# Hamming Window
frames *= np.hamming(frame_length)
c4 = dct(frames, type=2, axis=-1, norm='ortho')
plt.plot(c4)
plt.title('Windowing', fontsize=12, color='black')
plt.savefig(f'./out/output4_Windowing.png')
plt.show()

# Fourier-Transform and Power Spectrum
NFFT = 512
mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
c5 = dct(pow_frames, type=2, axis=-1, norm='ortho')
plt.plot(c5)
plt.title('DFT', fontsize=12, color='black')
plt.savefig(f'./out/output5_DFT.png')
plt.show()

# Filter Banks 滤波器
nfilt = 40
low_freq_mel = 0
high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
bin = np.floor((NFFT + 1) * hz_points / sample_rate)

fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
for m in range(1, nfilt + 1):
    f_m_minus = int(bin[m - 1])  # left
    f_m = int(bin[m])  # center
    f_m_plus = int(bin[m + 1])  # right

    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
filter_banks = np.dot(pow_frames, fbank.T)
filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
filter_banks = 20 * np.log10(filter_banks)  # dB

# Final Show
num_ceps = 12
c11 = dct(filter_banks, type=2, axis=-1, norm='ortho')[1: (num_ceps + 1)]  # Keep 2-13
plt.plot(c11)
plt.title('final1', fontsize=12, color='black')
plt.savefig(f'./out/output6_final1.png')
plt.show()

plt.imshow(np.flipud(filter_banks.T), cmap=plt.cm.jet, aspect=0.05,
           extent=[0, filter_banks.shape[1], 0, filter_banks.shape[0]])
plt.title('final2', fontsize=12, color='black')
plt.savefig(f'./out/output7_final2.png')
plt.show()
