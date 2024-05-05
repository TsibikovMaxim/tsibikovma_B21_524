import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def semitone(img):
    if str(img.mode) == "L":
        return img

    width = img.size[0]
    height = img.size[1]
    new_image = Image.new("L", (width, height))

    for x in range(width):
        for y in range(height):
            pix = img.getpixel((x, y))
            sum_ = 0.3 * pix[0] + 0.59 * pix[1] + 0.11 * pix[2]
            new_image.putpixel((x, y), int(sum_))
    return new_image


def haralic(img):
    side = 256

    img_matrix = np.asarray(img).transpose()
    width = img.size[0]
    height = img.size[1]
    matrix = np.zeros((side, side))
    for x in range(2, width - 2):
        for y in range(2, height - 2):
            pix = img_matrix[x, y]

            up_pix = img_matrix[x, y - 2]
            down_pix = img_matrix[x, y + 2]
            left_pix = img_matrix[x - 2, y]
            right_pix = img_matrix[x + 2, y]

            matrix[pix, up_pix] += 1
            matrix[pix, down_pix] += 1
            matrix[pix, left_pix] += 1
            matrix[pix, right_pix] += 1

    return Image.fromarray(matrix), matrix

def features(matrix):
    # AV (Average Value) и D (Deviation)
    width = matrix.shape[0]
    height = matrix.shape[1]

    av = np.sum(matrix) / (width * height)

    d = 0
    for i in range(0, height):
        for j in range(0, width):
            d += (matrix[i, j] - av) * (matrix[i, j] - av)

    return av, np.sqrt(d / (width * height))


def make_contrast(img_gray, gmin=0, gmax=255):
    width = img_gray.size[0]
    height = img_gray.size[1]

    result = np.zeros((width, height)).transpose()
    img_gray_arr = np.asarray(img_gray)

    fmin = img_gray_arr.min()
    fmax = img_gray_arr.max()

    for i in range(0, height):
        for j in range(0, width):
            a = (gmax - gmin) / (fmax - fmin)
            b = (gmin * fmax - gmax * fmin) / (fmax - fmin)
            result[i, j] = int(a * img_gray_arr[i, j] + b)

    return Image.fromarray(np.uint8(result)), result


def make_hist(matrix, save_path):
    sh = np.reshape(matrix, (1, -1))
    plt.figure()  # 0
    plt.hist(sh[0], bins=256)
    plt.savefig(save_path)


def features_and_contrast(img_path):
    name = img_path[-8:-4]

    img = Image.open(img_path).convert('RGB')
    img_gray = semitone(img)
    img_gray.save("output/images/" + name + "_gray.png")
    img_contrast, matrix_contrast = make_contrast(img_gray)

    img_haralic_gray, matrix_haralic_gray = haralic(img_gray)
    img_haralic_gray = img_haralic_gray.convert("L")
    img_haralic_gray.save(
        "output/images/" + name + "_gray_haralic.png")
    img_haralic_contrast, matrix_haralic_contrast = haralic(img_contrast)
    img_haralic_contrast = img_haralic_contrast.convert("L")
    img_haralic_contrast.save(
        "output/images/" + name + "_contrast.png")

    # создание гистограмм
    make_hist(np.asarray(img_gray),
              "output/images/" + name + "_gray_hist.png")
    make_hist(np.asarray(img_contrast),
              "output/images/" + name + "_contrast_hist.png")

    # подсчет свойст
    gray_features = features(matrix_haralic_gray / np.sum(matrix_haralic_gray))
    print("Свойства grayscale:")
    print(gray_features)
    contrast_features = features(matrix_haralic_contrast / np.sum(matrix_haralic_contrast))
    print("Свойства контрастного:")
    print(contrast_features)
