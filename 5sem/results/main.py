import utils

sin_letters = {
    '0D85', '0D86', '0D87', '0D88', '0D89', '0D8A', '0D8B', '0D8C',
    '0D8D', '0D8E', '0D8F', '0D90', '0D91', '0D92', '0D93', '0D94',
    '0D95', '0D96', '0D9A', '0D9B', '0D9C', '0D9D', '0D9E', '0D9F',
    '0DA0', '0DA1', '0DA2', '0DA3', '0DA4', '0DA7', '0DA8', '0DA9',
    '0DAA', '0DAB', '0DAC', '0DAD', '0DAE', '0DAF', '0DB0', '0DB1',
    '0DB3', '0DB4', '0DB5', '0DB6', '0DB7', '0DB8', '0DB9', '0DBA',
    '0DBB', '0DBD', '0DC0', '0DC5', '0DC1', '0DC2', '0DC3', '0DC4',
    '0DC6', '0DCA', '0DCF', '0DD0', '0DD1', '0DD2', '0DD3', '0DD4',
    '0DD6', '0DD8', '0DD9', '0DDA', '0DDB', '0DDC', '0DDD', '0DDE',
    '0DDF', '0DF2', '0DF3', '0DF4'
}

def main():
    # Генерация букв
    # utils.generate_letters(sin_letters)

    # Генерация фразы
    # utils.generate_phrase()

    # Инвертирование
    # utils.invert_letters(sin_letters)

    # Подсчет признаков
    # utils.create_profiles(sin_letters)
    utils.create_features(sin_letters)

if __name__ == "__main__":
    main()