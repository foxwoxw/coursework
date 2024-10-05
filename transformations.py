import numpy as np
import sys


# Функция масштабирования
# Принимает матрицу входного изображения (in_img), коэффициенты изменения масштаба по осям (Xfactor, Yfactor)
def scaling(in_img, Xfactor, Yfactor):
    height = in_img.shape[0]    # Высота входного изображения
    width = in_img.shape[1]     # Ширина входного изображения
    # Матрица преобразования 
    T = np.array([[Xfactor, 0], [0, Yfactor]], dtype = np.double)
    # Размерность выходного изображения с округлением до ближайшего целого 
    new_H = int(np.round(T[1,1] * height))  # Новая высота
    new_W = int(np.round(T[0,0] * width))   # Новая ширина
    # Проверка на допустимый размер для выходного изображения
    mult = new_H * new_W * in_img.ndim
    if mult > 1500*1500*3 or mult < 1:
        print("Недопустимое значение размера выходного изображения.")
        sys.exit(0)
    # Матрица выходного изображения
    out_img = np.zeros((new_H, new_W), dtype = np.double)
    return [out_img, T] 



# Формирование матрицы для результата поворота изображения
# angle - значение угла в градусах (отриц. - по часовой)
def rotate(in_img, angle):
    # Размер исходного изображения
    height = in_img.shape[0]    # Высота входного изображения
    width = in_img.shape[1]     # Ширина входного изображения
    # Проверка на допустимый размер для выходного изображения
    mult = height * width * in_img.ndim
    if mult > 1500*1500*3 or mult < 1:
        print("Недопустимое значение размера выходного изображения.")
        sys.exit(0)
    # Отсекаем повороты кратные 2pi
    angle %= 360
    # Значение угла в радианах
    radian = angle * (np.pi / 180)
    # Формирование матрицы прямого преобразования
    cos = np.cos(radian)
    sin = np.sin(radian)
    T = np.array([[cos, -sin], [sin, cos]], dtype = np.double)
    # Координаты центра входного изображения (начало координат)
    x0 = width / 2
    y0 = height / 2
    angle = abs(angle)
    # Вычисление размера выходной матрицы
    if angle > 0 and angle <= 90:
        max_row = y0 + (height - y0) * cos - (0 - x0) * sin
        min_row = y0 + (0 - y0) * cos - (width - x0) * sin
        max_column = x0 + (width - x0) * cos + (height - y0) * sin
        min_column = x0 + (0 - x0) * cos + (0 - y0) * sin
    elif angle > 90 and angle <= 180:
        max_row = y0 + (0 - y0) * cos - (0 - x0) * sin
        min_row = y0 + (height - y0) * cos - (width - x0) * sin
        max_column = x0 + (0 - x0) * cos + (height - y0) * sin
        min_column = x0 + (width - x0) * cos + (0 - y0) * sin
    elif angle > 180 and angle <= 270:
        max_row = y0 + (0 - y0) * cos - (width - x0) * sin
        min_row = y0 + (height - y0) * cos - (0 - x0) * sin
        max_column = x0 + (0 - x0) * cos + (0 - y0) * sin
        min_column = x0 + (width - x0) * cos + (height - y0) * sin
    else:
        max_row = y0 + (height - y0) * cos - (width - x0) * sin
        min_row = y0 + (0 - y0) * cos - (0 - x0) * sin
        max_column = x0 + (width - x0) * cos + (0 - y0) * sin
        min_column = x0 + (0 - x0) * cos + (height - y0) * sin
    # Формирование матрицы выходного изображения
    new_H = int(np.ceil(max_row - min_row + 1))
    new_W = int(np.ceil(max_column - min_column + 1))
    out_img = np.zeros((new_H, new_W), dtype = np.double)
    # Верхний левый угол
    x_left = min_column
    y_top = min_row
    out_img[0, 0] = x_left
    out_img[0, 1] = y_top
    return [out_img, T] 

