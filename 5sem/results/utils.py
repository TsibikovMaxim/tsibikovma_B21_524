import numpy as np
from math import ceil
from fontTools.ttLib import TTFont
from PIL import Image, ImageFont, ImageDraw
from PIL.ImageOps import invert
import csv
from matplotlib import pyplot as plt

class FontUtil:
    def __init__(self, font_path):
        self.font = TTFont(font_path)
        self.cmap = self.font['cmap']
        self.t = self.cmap.getcmap(3, 1).cmap
        self.s = self.font.getGlyphSet()
        self.units_per_em = self.font['head'].unitsPerEm

    def get_text_width(self, text, point_size):
        total = 0
        for c in text:
            if ord(c) in self.t and self.t[ord(c)] in self.s:
                total += self.s[self.t[ord(c)]].width
            else:
                total += self.s['.notdef'].width
        total = total * float(point_size)/self.units_per_em
        return total

def _semitone(old_img_arr):
    height = old_img_arr.shape[0]
    width = old_img_arr.shape[1]

    new_img_arr = np.zeros(shape=(height, width))

    for x in range(width):
        for y in range(height):
            pixel = old_img_arr[y, x]
            new_img_arr[y, x] = 0.3 * pixel[0] + 0.59 * pixel[1] + 0.11 * pixel[2]

    return new_img_arr.astype(np.uint8)

def _simple_binarization(old_image, threshold, semitone_needed=True):
    if semitone_needed:
        semi = _semitone(old_image)
    else:
        semi = old_image
    new_image = np.zeros(shape=semi.shape)

    new_image[semi > threshold] = 255

    return new_image.astype(np.uint8)

def generate_letters(sin_letters):
    font = ImageFont.truetype("input/notosans.ttf", 52)

    for i, letter_raw in enumerate(sin_letters):
        letter = chr(int(letter_raw, 16))
        width, height = font.getsize(letter)  # Get the width and height of the text

        img = Image.new(mode="RGB", size=(ceil(width), ceil(height)), color="white")

        draw = ImageDraw.Draw(img)

        draw.text((0, 0), letter, (0, 0, 0), font=font)

        img = Image.fromarray(_simple_binarization(np.array(img), 75),'L')
        img.save(f"output.txt/letters/{i+1}.png")

def generate_phrase():
    font = ImageFont.truetype("input/notosans.ttf", 52)

    width = 1000
    img = Image.new(mode="RGB", size=(ceil(width), 52), color="white")

    phrase = "මේක අවස්ථාවක්, විකල්පයක් නෙවෙයි"

    draw = ImageDraw.Draw(img)
    draw.text((0, 0), phrase, (0, 0, 0), font=font)

    img = Image.fromarray(_simple_binarization(np.array(img), 75),'L')
    img.save(f"output.txt/ex10.png")

def invert_letters(sin_letters):
    for i, _ in enumerate(sin_letters):
        img = Image.open(f"output/letters/{i+1}.png").convert('L')
        img = invert(img)
        img.save(f"output.txt/letters/inverse/{i+1}.png")

def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def last_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)

def get_profiles(img):
    return {
        'x': {
            'y': np.sum(img, axis=0),
            'x': np.arange(start=1, stop=img.shape[1] + 1).astype(int)
        },
        'y': {
            'y': np.arange(start=1, stop=img.shape[0] + 1).astype(int),
            'x': np.sum(img, axis=1)
        }
    }

def write_profile(img, iter, type='x'):
    profiles = get_profiles(img)

    if type == 'x':
        plt.bar(x=profiles['x']['x'], height=profiles['x']['y'], width=0.9)

        plt.ylim(0, 52)

    elif type == 'y':
        plt.barh(y=profiles['y']['y'], width=profiles['y']['x'], height=0.9)

        plt.ylim(52, 0)

    else:
        raise Exception('Unsupported profile')

    plt.xlim(0, 55)

    plt.savefig(f'output.txt/letters/profiles/{type}/{iter+1}.png')
    plt.clf()

def calculate_features(img):
    img_b = np.zeros(shape=img.shape)
    img_b[img != 255] = 1

    profiles = get_profiles(img_b)

    img_b = img_b[first_nonzero(profiles['y']['x'], 0): last_nonzero(
        profiles['y']['x'], 0) + 1, first_nonzero(profiles['x']['y'], 0): last_nonzero(profiles['x']['y'], 0) + 1]

    weight = img_b.sum()
    rel_weight = weight / (img_b.shape[0] * img_b.shape[1])

    x_avg = 0
    for x, column in enumerate(img_b.T):
        x_avg += sum((x + 1) * column)
    x_avg = x_avg/weight
    rel_x_avg = (x_avg-1)/(img_b.shape[1]-1)

    y_avg = 0
    for y, row in enumerate(img_b):
        y_avg += sum((y + 1) * row)
    y_avg = y_avg/weight
    rel_y_avg = (y_avg-1)/(img_b.shape[0]-1)

    iner_x = 0
    for y, row in enumerate(img_b):
        iner_x += sum((y + 1 - y_avg)**2 * row)
    rel_iner_x = iner_x/(img_b.shape[0]**2 + img_b.shape[1]**2)

    iner_y = 0
    for x, column in enumerate(img_b.T):
        iner_y += sum((x + 1 - x_avg)**2 * column)
    rel_iner_y = iner_y/(img_b.shape[0]**2 + img_b.shape[1]**2)

    return {
        'weight': weight, # Вес
        'rel_weight': rel_weight, # Удельный вес
        'center': (x_avg, y_avg), # Координаты центра тяжести
        'rel_center': (rel_x_avg, rel_y_avg), # Нормированные координаты центра тяжести
        'inertia': (iner_x, iner_y), # Нормированные осевые моменты инерции
        'rel_inertia': (rel_iner_x, rel_iner_y) # Профили X и Y
    }

def create_profiles(sin_letters):
   for i, letter_raw in enumerate(sin_letters):
        img_src = Image.open(f'output/letters/{i+1}.png').convert('L')
        img_src_arr = np.array(img_src)

        img_src_arr[img_src_arr == 0] = 1
        img_src_arr[img_src_arr == 255] = 0

        write_profile(img_src_arr, i)

def create_features(sin_letters):
    with open('output/data.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['letter', 'weight', 'rel_weight', 'center',
                                                     'rel_center', 'inertia', 'rel_inertia'])
        writer.writeheader()

        for i, letter_raw in enumerate(sin_letters):
            try:
                img_src = Image.open(f'output/letters/{i+1}.png').convert('L')
                img_src_arr = np.array(img_src)

                features = calculate_features(img_src_arr)
                features['letter'] = chr(int(letter_raw, 16))

                writer.writerow(features)
            except ZeroDivisionError:
                # Handle the ZeroDivisionError here, for example:
                print("ZeroDivisionError occurred for letter:", chr(int(letter_raw, 16)))
                # You can also choose to log the error or take other appropriate actions
                pass
