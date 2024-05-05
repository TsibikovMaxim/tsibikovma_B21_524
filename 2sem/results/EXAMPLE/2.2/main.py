import cv2
from glob import glob
import os

def cheating_otsu(input_path: str, output_path:str) -> None:
    """В рамках курса я буду использовать 
    в т.ч. методы бинаризации как модули 
    из сторонникх библиотек. Вам так делать нельзя!😜"""

    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Применение CLAHE
    # clahe = cv2.createCLAHE(clipLimit=img3, tileGridSize=(20,20))
    # image = clahe.apply(image)

    # Применение глобальной бинаризации с критерием Отсу
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)

    # Сохранение бинаризованного изображения в файлz    
    cv2.imwrite(output_path, binary_image)

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, 'output.txt')
    relative_path = "2sem/results/input/*"
    for input_path in glob(relative_path):
        curr_opath = os.path.join(output_path, os.path.basename(input_path))
        cheating_otsu(input_path, curr_opath)

if __name__ == "__main__":
    main()