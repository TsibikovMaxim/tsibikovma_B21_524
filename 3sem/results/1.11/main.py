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


def erosion(image, kernel):
    height, width = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    eroded_image = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            eroded_image[i, j] = np.min(padded_image[i:i + kernel_height, j:j + kernel_width] * kernel)

    return eroded_image


def dilation(image, kernel):
    height, width = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    dilated_image = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            dilated_image[i, j] = np.max(padded_image[i:i + kernel_height, j:j + kernel_width] * kernel)

    return dilated_image


def opening(image, kernel):
    eroded_image = erosion(image, kernel)
    opened_image = dilation(eroded_image, kernel)
    return opened_image

def difference_images(img1, img2):
    """Разница изображений."""
    result = ImageChops.difference(img1, img2)
    inv_img = ImageChops.invert(result)
    return inv_img

"""
Морфологическое открытие является комбинацией двух операций: эрозии, за которой следует дилатация. 
Это используется для удаления шума и мелких объектов из изображения, при этом сохраняя общую форму 
крупных объектов.

Шаг img1: Эрозия
Эрозия уменьшает объекты на изображении. Для каждого пикселя в исходном изображении:

Если все пиксели в структурном элементе совпадают с соответствующими пикселями изображения, 
пиксель остается неизменным.
В противном случае пиксель становится фоновым (устанавливается в 0).

Шаг img2: Дилатация
Дилатация увеличивает объекты на изображении. Для каждого пикселя в изображении после эрозии:

Если хотя бы один пиксель в структурном элементе совпадает с соответствующим пикселем изображения, 
пиксель устанавливается в img1.
В противном случае пиксель остается фоновым.
"""
def main():
    images = [
        "input/img1.png",
        "input/img2.png",
        "input/img3.png",
    ]
    for image in images:
        img_src = Image.open(image).convert('RGB')
        src_image = semitone(np.array(img_src))

        # Создание структурного элемента в виде квадрата 3x3
        structure = np.array([[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]], dtype=np.uint8)

        # Применение операции Opening
        opened_image = opening(src_image, structure)

        # Поиск разницы
        deffered_image = difference_images(
            Image.fromarray(src_image.astype(np.uint8), 'L').convert('RGB'),
            Image.fromarray(opened_image.astype(np.uint8), 'L').convert('RGB')
        )

        # Сохранение результата
        output_path = 'output.txt/img'+str(images.index(image)+1)+'/semitoned_image.png'
        plt.imsave(output_path, src_image, cmap='gray')

        output_path = 'output.txt/img'+str(images.index(image)+1)+'/opened_image.png'
        plt.imsave(output_path, opened_image, cmap='gray')

        output_path = 'output.txt/img'+str(images.index(image)+1)+'/diff_image.png'
        plt.imsave(output_path, deffered_image, cmap='gray')

if __name__ == "__main__":
    main()