from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np


def make_spectrogram(samples, sample_rate):
    frequencies, times, my_spectrogram = signal.spectrogram(samples, sample_rate, scaling='spectrum', window=('hann',))

    eps = np.finfo(float).eps
    my_spectrogram = np.maximum(my_spectrogram, eps)

    plt.pcolormesh(times, frequencies, np.log10(my_spectrogram), shading='auto')
    plt.ylabel('Частота [Гц]')
    plt.xlabel('Время [с]')

# Низких частот
def noise_reduction(samples, sample_rate, cutoff_freuency):
    z = signal.savgol_filter(samples, 101, 3)
    b, a = signal.butter(3, cutoff_freuency / sample_rate)
    zi = signal.lfilter_zi(b, a)
    z, _ = signal.lfilter(b, a, z, zi = zi * z[0])
    return z

def to_pcm(y):
    return np.int16(y / np.max(np.abs(y)) * 30000)