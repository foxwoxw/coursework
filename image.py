from PIL import Image
import numpy as np
import sys


# Запись изображения из файла в числовую матрицу
# filename - полный путь к файлу
def init_image(filename):
    try:
        with Image.open(filename) as img:
            if img.mode != 'L' or img.mode != 'P':
                in_img = img.convert('RGB')
            else:
                in_img = img.convert('L')
            in_img = np.array(in_img).astype(np.uint8)
        return in_img
    except FileNotFoundError:
        print('Файл не найден.')
        sys.exit(0)


# Сохранение изображения; img - числовая матрица
def save_img(img):
    print("Имя файла: ", end='')
    name = input()
    try:
        if img.ndim == 3:
            Image.fromarray((img),mode="RGB").save(name)
        else:
            Image.fromarray((img),mode="L").save(name)
    except:
        print("Ошибка сохранения. Сохранено как \"result.png\".")
        name = "result.png"
        if img.ndim == 3:
            Image.fromarray((img),mode="RGB").save(name)
        else:
            Image.fromarray((img),mode="L").save(name)



