import utils


def main():
    # Расчёта горизонтального и вертикального профиля изображения
    utils.create_profiles('x')
    utils.create_profiles('y')

    # Сегментация символов
    utils.symbol_selection()

    # Сегментация символов
    utils.text_selection()

if __name__ == "__main__":
    main()