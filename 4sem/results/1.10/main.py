from PIL import Image, ImageChops
import numpy as np
import matplotlib.pyplot as plt


def semitone(old_img_arr):
    height = old_img_arr.shape[0]
    width = old_img_arr.shape[1]

    new_img_arr = np.zeros(shape=(height, width))

    for x in range(width):
        for y in range(height):
            pixel = old_img_arr[y, x]
            new_img_arr[y, x] = 0.3 * pixel[0] + 0.59 * pixel[1] + 0.11 * pixel[2]

    return new_img_arr.astype(np.uint8)

def binarization(old_image, threshold):
    new_image = np.zeros(shape=old_image.shape)

    new_image[old_image > threshold] = 255

    return new_image.astype(np.uint8)

def pruitt_operator(image):
    # Оператор Прюитта для вычисления градиентов
    Gx = np.array([[-1, -1, -1, -1, -1],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1]])

    Gy = np.array([[-1, 0, 0, 0, 1],
                   [-1, 0, 0, 0, 1],
                   [-1, 0, 0, 0, 1],
                   [-1, 0, 0, 0, 1],
                   [-1, 0, 0, 0, 1]])

    # Получение размеров изображения
    height, width = image.shape

    # Инициализация градиентных матриц
    Gx_result = np.zeros_like(image, dtype=np.float32)
    Gy_result = np.zeros_like(image, dtype=np.float32)

    # Вычисление градиентов
    for y in range(2, height - 2):
        for x in range(2, width - 2):
            window = image[y - 2:y + 3, x - 2:x + 3]
            Gx_result[y, x] = np.sum(Gx * window)
            Gy_result[y, x] = np.sum(Gy * window)

    # Вычисление общей градиентной матрицы G
    G_result = np.abs(Gx_result) + np.abs(Gy_result)

    # Нормализация значений яркости
    G_result = ((G_result - np.min(G_result)) / (np.max(G_result) - np.min(G_result))) * 255

    return (Gx_result.astype(np.uint8),
            Gy_result.astype(np.uint8),
            G_result.astype(np.uint8))

def main():
    images = [
        "input/img1.png",
        "input/img2.png",
        "input/img3.png",
    ]
    for image in images:
        img_src = Image.open(image).convert('RGB')
        src_image = semitone(np.array(img_src))

        pruitt_x_image, pruitt_y_image, pruitt_image = pruitt_operator(src_image)

        binarized_100_image = binarization(src_image, 50)
        binarized_150_image = binarization(src_image, 150)
        binarized_200_image = binarization(src_image, 200)

        # Сохранение результата
        output_path = 'output.txt/img' + str(images.index(image) + 1) + '/semitoned_image.png'
        plt.imsave(output_path, src_image, cmap='gray')

        output_path = 'output.txt/img' + str(images.index(image) + 1) + '/pruitt_x.png'
        plt.imsave(output_path, pruitt_x_image, cmap='gray')

        output_path = 'output.txt/img' + str(images.index(image) + 1) + '/pruitt_y.png'
        plt.imsave(output_path, pruitt_y_image, cmap='gray')

        output_path = 'output.txt/img' + str(images.index(image) + 1) + '/pruitt.png'
        plt.imsave(output_path, pruitt_image, cmap='gray')

        output_path = 'output.txt/img' + str(images.index(image) + 1) + '/binarized_100.png'
        plt.imsave(output_path, binarized_100_image, cmap='gray')

        output_path = 'output.txt/img' + str(images.index(image) + 1) + '/binarized_150.png'
        plt.imsave(output_path, binarized_150_image, cmap='gray')

        output_path = 'output.txt/img' + str(images.index(image) + 1) + '/binarized_200.png'
        plt.imsave(output_path, binarized_200_image, cmap='gray')


if __name__ == "__main__":
    main()