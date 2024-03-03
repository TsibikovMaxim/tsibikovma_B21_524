from PIL import Image
import numpy as np


def stretching(img_src_arr, m):
    height = img_src_arr.shape[0]
    width = img_src_arr.shape[1]
    new_width = int(width * m)
    new_height = int(height * m)

    new_image_arr = np.zeros(shape=(new_height, new_width, img_src_arr.shape[2]))

    for x in range(new_width - 1):
        for y in range(new_height - 1):
            new_image_arr[x, y] = img_src_arr[int(x / m), int(y / m)]

    return new_image_arr


def reducing(img_src_arr, n):
    height = img_src_arr.shape[0]
    width = img_src_arr.shape[1]
    new_width = int(width / n)
    new_height = int(height / n)

    new_image_arr = np.zeros(shape=(new_height, new_width, img_src_arr.shape[2]))

    for x in range(new_width):
        for y in range(new_height):
            new_image_arr[x, y] = img_src_arr[int(x * n), int(y * n)]

    return new_image_arr


def oversampling_twostep(img_src_arr, m, n):
    return reducing(stretching(img_src_arr, m), n)


def oversampling(img_src_arr, scale):
    height = img_src_arr.shape[0]
    width = img_src_arr.shape[1]
    new_height = round(scale * width)
    new_width = round(scale * width)

    new_image = np.zeros(shape=(new_height, new_width, img_src_arr.shape[2]))

    for x in range(new_width):
        for y in range(new_height):
            src_x = int(float(x) / float(new_width) * float(width))
            src_y = int(float(y) / float(new_height) * float(height))

            new_image[y, x] = img_src_arr[src_y, src_x]

    return new_image


def main():
    img_src = Image.open("input/img.png").convert('RGB')
    img_src_arr = np.array(img_src)

    # 1.1
    src_image_stretched = stretching(img_src_arr, 2)
    img = Image.fromarray(src_image_stretched.astype(np.uint8), 'RGB')
    img.save("output/stretched.png")

    # 1.2
    src_image_reduced = reducing(img_src_arr, 2)
    img = Image.fromarray(src_image_reduced.astype(np.uint8), 'RGB')
    img.save("output/reduced.png")

    # 1.3
    src_image_oversampled_twostep = oversampling_twostep(img_src_arr, 7, 2)
    img = Image.fromarray(src_image_oversampled_twostep.astype(np.uint8), 'RGB')
    img.save("output/oversampled_twostep.png")

    # 1.4
    src_image_oversampled = oversampling(img_src_arr, 3.5)
    img = Image.fromarray(src_image_oversampled.astype(np.uint8), 'RGB')
    img.save("output/oversampled.png")


if __name__ == "__main__":
    main()
