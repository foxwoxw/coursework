from PIL import Image
import numpy as np


# функция для вычисления ядра Ланцоша
def lanczos_kernel(x, a):
    if x == 0:
        return 1.0
    elif -a < x < a:
        return (a * np.sin(np.pi * x) * np.sin(np.pi * x / a)) / (np.pi * np.pi * x * x)
    else:
        return 0.0


# Функция для вычисления 2D ядра Ланцоша
def lanczos_kernel_2d(x, y, a):
    return lanczos_kernel(x, a) * lanczos_kernel(y, a)# Умножаем 1D ядра по осям x и y




# Функция интерполяции с использованием ядра Ланцоша
def interpolate(in_img, w, h, a, height, width):
    # Нахождение ближайших целых координат пикселей
    x1 = int(np.floor(w))
    y1 = int(np.floor(h))
    color = 0  # Для хранения значений цвета 
    total_weight = 0    # Переменная для накопления веса
    # Окрестность пикселя
    min_a = -a + 1
    max_a = a + 1
    # Проходим по окрестности пикселя
    for dx in range(min_a, max_a):
        delta_x = (x1 + dx)
        # Если не выходит за границы - считаем ядро по ширине
        if delta_x >= 0 and delta_x < width:
            kernel_x = lanczos_kernel(w - delta_x, a)
        else:
            continue
        for dy in range(min_a, max_a):
            delta_y = (y1 + dy)
            # Проверяем, не выходит ли индекс за пределы изображения
            if delta_y >= 0 and delta_y < height:                
                # Получаем цвет пикселя
                pixel = in_img[delta_y, delta_x]
                # Вычисляем вес для текущего пикселя
                kernel_y = lanczos_kernel(h - delta_y, a)
                # 2D ядро Ланцоша
                weight = kernel_x * kernel_y
                # Увеличиваем значения цвета с учетом веса
                color += pixel * weight
                # Накопление общего веса
                total_weight += weight
    # Если общий вес больше нуля, нормализуем цвета
    if total_weight > 0:
        color /= total_weight
    if color > 255: color = 255
    elif color < 0: color = 0
    else: color = int(color)
    return color


# Интерполяция Ланцоша для оодного канала (Масштабирование)
def lanczos_one_channel_scale(in_image, out_img, Xfactor, Yfactor, a):
    height, width = in_image.shape[0], in_image.shape[1]    # Размер исходного изображения
    new_H, new_W = out_img.shape[0], out_img.shape[1]   # Размер выходного изображения
    # Проходимся по каждому пикселю нового изображения
    for y in range(new_H):
        for x in range(new_W):
            # Вычисление исходных коорддинат
            w = x / Xfactor
            h = y / Yfactor
            # Вычисление цвета пикселей
            color = interpolate(in_image, w, h, a, height, width)
            # Заполнение матрицы нового изображения
            out_img[y, x] =  color
    out_img = out_img.astype(np.uint8)
    return out_img



# Интерполяция Ланцоша для оодного канала (поворот)
def lanczos_one_channel_rotate(in_img, out_img, cos_rad, sin_rad, x_left, y_top, a):
    height, width = in_img.shape[0], in_img.shape[1]    # Размер исходного изображения
    new_H, new_W = out_img.shape[0], out_img.shape[1]   # Размер выходного изображения
    # Центр исходного изображения
    h0 = height / 2
    w0 = width / 2
    # Проходим по каждому пикселю выходного изображения
    for y in range(new_H):
        # Смещение начала координат по высоте
        delta_y = y + y_top - h0
        # Константы относительно цикла по ширине
        delta_y_cos = delta_y * cos_rad
        delta_y_sin = delta_y * sin_rad
        for x in range(new_W):
            # Смещение начала координат по ширине
            delta_x = x + x_left - w0
            # Обратное преобразование координат
            h = delta_y_cos - delta_x * sin_rad + h0
            w = delta_y_sin + delta_x * cos_rad + w0
            # Проверяем, находятся ли оригинальные координаты в пределах входного изображения
            if 0 <= w < width and 0 <= h < height:
                # Интерполируем значение пикселя с использованием метода Ланцоша
                color = interpolate(in_img, w, h, a, height, width)
                out_img[y, x] = color
    out_img = out_img.astype(np.uint8)
    return out_img


# Главная функция
def lanczos_interpolation(in_img, out_img, T, action, a):
    # Каждый цветовой канал обрабатывается отдельно
    if in_img.ndim == 3:
        temp = []
        if action == '2':  # Масштабирование
            Xfactor = T[0, 0]
            Yfactor = T[1, 1]
            for i in range(0, 3):
                temp.append(lanczos_one_channel_scale(in_img[:,:,i], out_img, Xfactor, Yfactor, a))
                print(i)
        else:  # Поворот
            sin_rad = T[0, 1]
            cos_rad = T[0, 0]
            x_left, y_top = out_img[0, 0], out_img[0, 1]
            # При создании матрицы out_img ячейки [0,0] и [0,1] заполнялись ненулевыми значениями.
            # Обнуляем их, чтобы в результате использовать нулевые пиксели как черный фон.
            out_img[0,0], out_img[0,1] = 0, 0
            for i in range(0, 3):
                temp.append(lanczos_one_channel_rotate(in_img[:,:,i], out_img, cos_rad, sin_rad, x_left, y_top, a))
        out_img = np.stack((temp[0], temp[1], temp[2]), axis=2)
        Image.fromarray((out_img),mode="RGB").show()

    else:
        if action == '2':  # Масштабирование
            Xfactor = T[0, 0]
            Yfactor = T[1, 1]
            out_img = lanczos_one_channel_scale(in_img, out_img, Xfactor, Yfactor, a)
        else:  # Поворот
            sin_rad = T[0, 1]
            cos_rad = T[0, 0]
            x_left, y_top = out_img[0, 0], out_img[0, 1]
            # При создании матрицы out_img ячейки [0,0] и [0,1] заполнялись ненулевыми значениями.
            # Обнуляем их, чтобы в результате использовать нулевые пиксели как черный фон.
            out_img[0,0], out_img[0,1] = 0, 0
            out_img = lanczos_one_channel_rotate(in_img, out_img, cos_rad, sin_rad, x_left, y_top, a)
        Image.fromarray((out_img), mode="L").show()
    return out_img
