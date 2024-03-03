from PIL import Image
import numpy as np


def semitone(old_img_arr):
    height = old_img_arr.shape[0]
    width = old_img_arr.shape[1]

    new_img_arr = np.zeros(shape=(height, width))

    for x in range(width):
        for y in range(height):
            pixel = old_img_arr[y, x]
            new_img_arr[y, x] = 0.3 * pixel[0] + 0.59 * pixel[1] + 0.11 * pixel[2]

    return new_img_arr.astype(np.uint8)


def main():
    img_src = Image.open("../input/198_115.png").convert('RGB')
    img_src_arr = np.array(img_src)

    src_image = semitone(img_src_arr)
    img = Image.fromarray(src_image, 'L').convert('RGB')
    img.save("output/semitoned.png")


if __name__ == "__main__":
    main()
