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
    return my_spectrogram, frequencies


# Получение интегральной матрицы
def integral(spectogram):
    height = spectogram.shape[0]
    width = spectogram.shape[1]

    integral_img_arr = np.zeros(shape=(height, width))

    for y in range(height):
        for x in range(width):
            pixel = spectogram[y, x]
            if x == 0 and y == 0:
                new_pixel = pixel
            elif x == 0 and y > 0:
                new_pixel = pixel + integral_img_arr[y - 1][x]
            elif x > 0 and y == 0:
                new_pixel = pixel + integral_img_arr[y][x - 1]
            else:
                new_pixel = pixel - integral_img_arr[y - 1][x - 1] + integral_img_arr[y - 1][x] + integral_img_arr[y][
                    x - 1]

            integral_img_arr[y][x] = new_pixel

    return integral_img_arr


# средняя энергия в точке с некоторой окрестностью
def brightness(integral_arr, x, y, aperture=7):
    height = integral_arr.shape[0]
    width = integral_arr.shape[1]

    start = (max(x - aperture, 0), max(y - aperture, 0))
    end = (min(x + aperture, width - 1), min(y + aperture, height - 1))

    sum_region = (
            integral_arr[end[1]][end[0]]
            - integral_arr[end[1]][start[0]]
            - integral_arr[start[1]][end[0]]
            + integral_arr[start[1]][start[0]]
    )

    square = (end[0] - start[0]) * (end[1] - start[1])

    # Обрабатываем случай, когда square равно нулю
    if square == 0:
        return 0

    return sum_region / square


# 5 частот(формант) с наибольшей энергией в какой-то момент (по возрастанию энергии на каждой частоте)
# возвращает список частот с энергией при этой частоте
def formant_moment(frequencies, integral_arr, x, aperture):
    # Вычисляем яркость для каждой частоты
    brightness_values = [brightness(integral_arr, x, i, aperture) for i in range(1, integral_arr.shape[0], 3)]

    # Находим индексы 5 частот с наибольшей энергией
    top_indices = sorted(range(len(brightness_values)), key=lambda i: brightness_values[i])[-5:]

    # Сортируем частоты по возрастанию энергии, исключая NaN
    sorted_formants = sorted(
        [(frequencies[i], brightness_values[i]) for i in top_indices if not np.isnan(brightness_values[i])],
        key=lambda x: x[1])

    # Возвращаем частоты и их энергии
    return [int(formant) for formant, _ in sorted_formants], [(int(formant), int(energy)) for formant, energy in
                                                              sorted_formants]

#Нахождение всех формант
def all_formants(frequencies, integral_arr, aperture):
    res = set()
    for i in range(integral_arr.shape[1]):
        formant = formant_moment(frequencies, integral_arr, i, aperture)[0]
        if formant[1] == 0:
            continue
        else:
            for j in range(5):
                res.add(formant[j])
    res.discard(0)

    return res

def power(frequencies, integral_arr, aperture, formant_s):
    power_s = dict()
    for i in formant_s:
        power_s[i] = 0

    for i in range(integral_arr.shape[1]):
        formant = formant_moment(frequencies, integral_arr, i, aperture)[0]
        if formant[1] == 0: #грубо, но сойдет
            continue
        else:
            for j in formant_moment(frequencies, integral_arr, i, 1)[1]:
                if (j[0] != 0):
                    power_s[j[0]] += j[1]

    return power_s