from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import utils

def main():
    size = 1000

    sample_rate_a, samples_a = wavfile.read("input/A.wav")
    sample_rate_i, samples_i = wavfile.read("input/I.wav")
    sample_rate_woof, samples_woof = wavfile.read("input/Woof.wav")

    plt.figure(dpi=size)

    spectogram_a, frequencies_a = utils.make_spectrogram(samples_a, sample_rate_a)
    plt.savefig('output/spectrogram_A.png', dpi=size)

    spectogram_i, frequencies_i = utils.make_spectrogram(samples_i, sample_rate_i)
    plt.savefig('output/spectrogram_I.png', dpi=size)

    spectogram_woof, frequencies_woof = utils.make_spectrogram(samples_woof, sample_rate_woof)
    plt.savefig('output/spectrogram_Woof.png', dpi=size)

    spec_integral_a = utils.integral(spectogram_a)
    formants_a = utils.formant_moment(frequencies_a, spec_integral_a, 1439, 1)
    print("Форманты для 'А':", formants_a)

    spec_integral_i = utils.integral(spectogram_i)
    formants_i = utils.formant_moment(frequencies_i, spec_integral_i, 900, 1)
    print("Форманты для 'И':", formants_i)

    spec_integral_woof = utils.integral(spectogram_woof)
    formants_woof = utils.formant_moment(frequencies_woof, spec_integral_woof, 800, 1)
    print("Форманты для 'Woof':", formants_woof)

    print("\n")

    # Найти минимальную и максимальную частоту голоса
    formants_a_freqs, _ = utils.formant_moment(frequencies_a, spec_integral_a, 1400, 1)
    formants_i_freqs, _ = utils.formant_moment(frequencies_i, spec_integral_i, 900, 1)
    formants_woof_freqs, _ = utils.formant_moment(frequencies_woof, spec_integral_woof, 800, 1)

    min_freq_a, max_freq_a = min(formants_a_freqs), max(formants_a_freqs)
    min_freq_i, max_freq_i = min(formants_i_freqs), max(formants_i_freqs)
    min_freq_woof, max_freq_woof = min(formants_woof_freqs), max(formants_woof_freqs)

    print("Минимальная частота для 'А':", min_freq_a)
    print("Максимальная частота для 'А':", max_freq_a)

    print("------------------------------------")

    print("Минимальная частота для 'И':", min_freq_i)
    print("Максимальная частота для 'И':", max_freq_i)

    print("------------------------------------")

    print("Минимальная частота для 'Woof':", min_freq_woof)
    print("Максимальная частота для 'Woof':", max_freq_woof)

if __name__ == "__main__":
    main()

# Форманты для 'А': ([7125, 7312, 7500, 7687, 7875], [(7125, 0), (7312, 0), (7500, 0), (7687, 0), (7875, 0)])
# Форманты для 'И': ([562, 3937, 187, 375, 0], [(562, 27), (3937, 39), (187, 275), (375, 390), (0, 1202)])
# Форманты для 'Woof': ([3750, 937, 375, 562, 0], [(3750, 4650), (937, 4668), (375, 5531), (562, 5916), (0, 7175)])
#
#
# Минимальная частота для 'А': 0
# Максимальная частота для 'А': 750
# ------------------------------------
# Минимальная частота для 'И': 0
# Максимальная частота для 'И': 3937
# ------------------------------------
# Минимальная частота для 'Woof': 0
# Максимальная частота для 'Woof': 3750