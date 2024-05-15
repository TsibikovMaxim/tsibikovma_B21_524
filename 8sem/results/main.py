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
    print(attributes) # (1.52587890625e-05, 0.0019463854472921867)

    # # Контрастирование
    utils.make_hist(matrix_res, "output/img2.png")

    # Объединение
    utils.features_and_contrast("input/img1.png")
    # Свойства grayscale:
    # (1.52587890625e-05, 0.0019463854472921867)
    # Свойства контрастного:
    # (1.52587890625e-05, 0.0019463854472921867)

    utils.features_and_contrast("input/img2.png")
    # Свойства grayscale:
    # (1.52587890625e-05, 2.579468089691907e-05)
    # Свойства контрастного:
    # (1.52587890625e-05, 2.579468089691907e-05)

    utils.features_and_contrast("input/img3.png")
    # Свойства grayscale:
    # (1.5258789062499998e-05, 0.00016813812400894623)
    # Свойства контрастного:
    # (1.5258789062499998e-05, 0.00016813812400894623)

    utils.features_and_contrast("input/img4.png")
    # Свойства grayscale:
    # (1.5258789062500003e-05, 0.00023233235660204447)
    # Свойства контрастного:
    # (1.5258789062500003e-05, 0.00023233235660204447)

if __name__ == "__main__":
    main()