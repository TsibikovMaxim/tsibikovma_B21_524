from PIL import Image
import numpy as np
import utils


def main():
    img = np.array(Image.open("input/sentence1.png").convert('L'))

    # Подсчет признаков
    # features = utils.calculate_features(img)
    # print(features)

    # Сегментация текста
    # borders = utils.get_symbol_boxes(img)
    # print(borders)

    path = 'input/data.csv'
    # Классификация
    # features = utils.load_features(path)
    # print(features)
    utils.text_recognition(
        path = path,
        phrase_path='input/sentence1.png',
        features_result_path="output/output.txt")

    utils.text_recognition(
        path=path,
        phrase_path='input/sentence2.png',
        features_result_path="output/output.txt")

    utils.text_recognition(
        path=path,
        phrase_path='input/sentence3.png',
        features_result_path="output/output.txt")

    utils.text_recognition(
        path=path,
        phrase_path='input/sentence4.png',
        features_result_path="output/output.txt")

    utils.text_recognition(
        path=path,
        phrase_path='input/sentence4_enlarged.png',
        features_result_path="output/output.txt")


if __name__ == "__main__":
    main()