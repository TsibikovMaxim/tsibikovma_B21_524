import utils
from PIL import Image
import numpy as np


def main():
    # Текстурный анализ
    img = Image.open("input/img1.png").convert('RGB')
    img_gray = utils.semitone(img)
    img_res, matrix_res = utils.haralic(img_gray)
    img_res = img_res.convert("L")
    img_res.save(f"output/img1.png")

    k = np.sum(matrix_res)
    norm_matrix_res = matrix_res / k

    attributes = utils.features(norm_matrix_res)
    print(attributes) # (47.33759821677704, 0.6582151456661353)

    # # Контрастирование
    utils.make_hist(matrix_res, "output/img2.png")

    # Объединение
    utils.features_and_contrast("input/img1.png")
    # Свойства grayscale:
    # (47.33759821677704, 0.6582151456661353)
    # Свойства контрастного:
    # (47.33759821677704, 0.6582151456661353)

    utils.features_and_contrast("input/img2.png")
    # Свойства grayscale:
    # (2849.2947300780734, 0.02691102189744494)
    # Свойства контрастного:
    # (2849.2947300780734, 0.02691102189744494)

    utils.features_and_contrast("input/img3.png")
    # Свойства grayscale:
    # (172.23121619157007, 0.2243533935269305)
    # Свойства контрастного:
    # (282.792590860763, 0.20210149969103533)

    utils.features_and_contrast("input/img4.png")
    # Свойства grayscale:
    # (1767.9931511310215, 0.14387425404478726)
    # Свойства контрастного:
    # (1767.9931511310215, 0.14387425404478726)

if __name__ == "__main__":
    main()