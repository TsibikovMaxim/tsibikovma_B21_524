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


def adaptive_binarization(image_array, window_size=15, c=10):
    # Ensure window_size is odd to have a center pixel
    if window_size % 2 == 0:
        window_size += 1

    height, width = image_array.shape
    binarized_image = np.zeros_like(image_array)

    # Pad the image to handle the borders
    padded_image = np.pad(image_array, window_size // 2, mode='edge')

    # Iterate over the image
    for y in range(height):
        for x in range(width):
            # Extract the local region
            local_region = padded_image[y:y + window_size, x:x + window_size]
            # Compute the local threshold (mean - c)
            local_threshold = np.mean(local_region) - c

            # Apply the threshold
            if image_array[y, x] > local_threshold:
                binarized_image[y, x] = 255
            else:
                binarized_image[y, x] = 0

    return binarized_image


def main():
    images = [
        "input/img1.png",
        "input/img2.png",
        "../input/198_115.png"
    ]

    for i in range(len(images)):
        img_src = Image.open(images[i]).convert('RGB')
        img_src_arr = np.array(img_src)

        gray_img_arr = semitone(img_src_arr)  # Use your `semitone` function here

        # Then, apply adaptive binarization
        binarized_image = adaptive_binarization(gray_img_arr)

        # Finally, save the binarized image
        img = Image.fromarray(binarized_image)
        path = "output/semitoned_" + str(i+1) + ".png"
        img.save(path)


if __name__ == "__main__":
    main()