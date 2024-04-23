from PIL import Image
from PIL.ImageOps import invert
import numpy as np
import matplotlib.pyplot as plt
import os

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

def write_profile(img, type):
    profiles = get_profiles(img)


    if type == 'x':
        plt.bar(x=profiles['x']['x'], height=profiles['x']['y'], width=0.9)

        plt.ylim(0, 100)

    elif type == 'y':
        plt.barh(y=profiles['y']['y'], width=profiles['y']['x'], height=0.9)

        plt.ylim(40, 0)

    else:
        raise Exception('Unsupported profile')

    plt.xlim(0, 500)

    plt.savefig(f'output/profiles/profiles_{type}.png')

def create_profiles(type):
    img_src = Image.open(f'input/sentence.png').convert('L')
    img_src_arr = np.array(img_src)

    img_src_arr[img_src_arr <= 75] = 1
    img_src_arr[img_src_arr > 75] = 0

    write_profile(img_src_arr, type=type)

def calculate_profiles(img):
    return {
        'x': np.sum(img, axis=0).astype(int),
        'y': np.sum(img, axis=1).astype(int)
    }

def get_symbol_boxes(img):
    profiles = calculate_profiles(img)
    borders = []

    i = 0
    while i < profiles['x'].shape[0]:
        current = profiles['x'][i]
        if current != 0:
            x1 = i
            count = 0
            while i + count < profiles['x'].shape[0] and profiles['x'][i + count] != 1:
                count += 1
            i += count
            x2 = i
            borders.append((x1, x2))
        i += 1

    return borders

def semitone(old_img_arr):
    height = old_img_arr.shape[0]
    width = old_img_arr.shape[1]

    new_img_arr = np.zeros(shape=(height, width))

    for x in range(width):
        for y in range(height):
            pixel = old_img_arr[y, x]
            new_img_arr[y, x] = 0.3 * pixel[0] + 0.59 * pixel[1] + 0.11 * pixel[2]

    return new_img_arr.astype(np.uint8)

def simple_binarization(old_image, threshold, semitone_needed=True):
    if semitone_needed:
        semi = semitone(old_image)
    else:
        semi = old_image
    new_image = np.zeros(shape=semi.shape)

    new_image[semi > threshold] = 255

    return new_image.astype(np.uint8)

def symbol_selection():
    img_src = Image.open(f'input/sentence.png').convert('L')
    img_src_arr = np.array(img_src)

    img_arr = np.zeros(shape=img_src_arr.shape)
    img_arr[img_src_arr < 75] = 1
    img_arr[img_src_arr > 75] = 0

    for i, (x1, x2) in enumerate(get_symbol_boxes(img_arr)):
        invert(Image.fromarray(img_src_arr[:, x1:x2])).save(
            f"output/symbols/{i + 1}.png"
        )

def calculate_profiles(img):
    profile_x = np.sum(img, axis=0)
    profile_y = np.sum(img, axis=1)

    return {
        'x': profile_x,
        'y': profile_y
    }

def show_profile(img, type='x'):
    if type == 'x':
        profile_y = calculate_profiles(img)[type]
        profile_x = np.arange(
            start=1, stop=img.shape[1] + 1).astype(int)

        plt.bar(x=profile_x, height=profile_y, width=0.9)

        plt.ylim(0, img.shape[0])

    elif type == 'y':
        profile_y = calculate_profiles(img)[type]
        profile_x = np.arange(
            start=1, stop=img.shape[0] + 1).astype(int)

        plt.barh(y=profile_x, width=profile_y, height=0.9)

        plt.ylim(52, 0)

    else:
        raise Exception('Unsupported profile')

    plt.xlim(0, img.shape[1])


def get_text_box(img, h_gap, v_gap):
    profiles = calculate_profiles(img)

    x1, x2, y1, y2 = None, None, None, None
    i = 0
    while i < profiles['x'].shape[0]:
        current = profiles['x'][i]
        if current != 0 and x1 == None:
            x1 = i
        elif current == 0:
            if x1 == None:
                pass
            else:
                count = 0
                while profiles['x'][i + count] == 0:
                    if count == h_gap:
                        x2 = i
                        i = profiles['x'].shape[0]
                        break
                    if i + count >= profiles['x'].shape[0] - 1:
                        x2 = i
                        i = profiles['x'].shape[0]
                        break
                    count += 1
                i += count
                continue
        i += 1
    if x2 == None:
        x2 = i

    i = 0
    while i < profiles['y'].shape[0]:
        current = profiles['y'][i]
        if current != 0 and y1 == None:
            y1 = i
        elif current == 0:
            if y1 == None:
                pass
            else:
                count = 0
                while profiles['y'][i + count] == 0:
                    if count == v_gap:
                        y2 = i
                        count += 1
                        break
                    if i + count >= profiles['y'].shape[0] - 1:
                        y2 = i
                        count += 1
                        break
                    count += 1
                i += count
                continue
        i += 1
    if y2 == None:
        y2 = i

    return (x1, y1), (x2, y2)

def text_selection():
    img_src = Image.open(f'input/sentence.png').convert('L')
    invert(img_src).save("output/inverted_sentence.png")
    img_src_arr = np.array(img_src)

    img_arr = np.zeros(shape=img_src_arr.shape)
    img_arr[img_src_arr == 0] = 1
    img_arr[img_src_arr == 255] = 0

    (x1, y1), (x2, y2) = get_text_box(img_arr, h_gap=17, v_gap=52)

    invert(Image.fromarray(img_src_arr[y1:y2, x1:x2])).save(
        "output/selected_sentence.png"
    )

    print((x1, x2), (y1, y2))

    plt.bar(x=x1, height=y1, width=0.9)
    plt.savefig(f'output/selected_sentence_profile1.png')

    plt.bar(x=x2, height=y2, width=0.9)
    plt.savefig(f'output/selected_sentence_profile2.png')

