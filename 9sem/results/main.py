from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import utils


def main():
    size = 500

    # Построение спектрограммы
    sample_rate, samples = wavfile.read("input/sound1.wav")
    plt.figure(dpi=size)
    utils.make_spectrogram(samples, sample_rate)
    plt.savefig('output/spectrogram_original.png', dpi=size)

    # Оценка уровня шума и его удаление
    reduced_noise = utils.noise_reduction(samples, sample_rate, cutoff_freuency=4000)
    utils.make_spectrogram(reduced_noise, sample_rate)
    plt.savefig('output/spectrogram_noise_reduced.png', dpi=size)

    # Восстановление звуковой дорожки
    wavfile.write('output/reduced_noise.wav', sample_rate, utils.to_pcm(reduced_noise))
    reduced_twice_noise = utils.noise_reduction(reduced_noise, sample_rate, cutoff_freuency=4000)
    utils.make_spectrogram(reduced_twice_noise, sample_rate)
    plt.savefig('output/spectrogram_twice_noise_reduced.png', dpi=size)
    wavfile.write('output/reduced_twice_noise.wav', sample_rate, utils.to_pcm(reduced_twice_noise))

    # Поиск моментов времени с наибольшей энергией
    frequencies, times, my_spectrogram = signal.spectrogram(reduced_twice_noise, sample_rate, scaling='spectrum',
                                                            window=('hann',))
    energies = np.sum(my_spectrogram, axis=0)
    peaks, _ = signal.find_peaks(energies, distance=1)
    # Отображение моментов времени с наибольшей энергией
    plt.figure()
    plt.plot(times, energies)
    plt.plot(times[peaks], energies[peaks], "x")
    plt.xlabel('Время [с]')
    plt.ylabel('Энергия')
    plt.title('Моменты с наибольшей энергией')
    plt.savefig('output/high_energy_moments.png')


if __name__ == "__main__":
    main()
