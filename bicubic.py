import numpy as np
from PIL import Image


# Обратная матрица коэффициентов при a_ij
invH = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [-3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0],
                 [-3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0],
                 [9, -9, -9, 9, 6, 3, -6, -3, 6, -6, 3, -3, 4, 2, 2, 1], 
                 [-6, 6, 6, -6, -3, -3, 3, 3, -4, 4, -2, 2, -2, -2, -1, -1],
                 [2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                 [-6, 6, 6, -6, -4, -2, 4, 2, -3, 3, -3, 3, -2, -1, -2, -1],
                 [4, -4, -4, 4, 2, 2, -2, -2, 2, -2, 2, -2, 1, 1, 1, 1]], dtype = np.double)



# Вычисление производных в конечных разностях
def derivative_calculation(in_img):
    # Размер исходного изображения
    height, width = in_img.shape[0], in_img.shape[1]
    # Производные по горизонтали
    f_x = np.zeros((height, width), dtype = np.double)
    width -= 1
    for B in range(0, height):
        for A in range(1, width):
            f_x[B, A] = (in_img[B, A + 1] - in_img[B, A - 1]) / 2.0
    width += 1
    # Производные по вертикали
    f_y = np.zeros((height, width), dtype = np.double)
    height -= 1
    for B in range(1, height):
        for A in range(0, width):
            f_y[B, A] = (in_img[B + 1, A] - in_img[B - 1, A]) / 2.0
    # Смешанные производные
    width -= 1
    f_xy = np.zeros((height + 1, width + 1), dtype = np.double)
    for B in range(1, height):
        for A in range(1, width):
            f_xy[B, A] = (in_img[B - 1, A - 1] + in_img[B + 1, A + 1] - in_img[B - 1, A + 1] - in_img[B + 1, A - 1]) / 4.0
    return [f_x, f_y, f_xy]



# Обработка одного цветового канала (масштабирование)
def bicubic_one_channel_scale(in_img, out_img, Xfactor, Yfactor):
    height, width = in_img.shape[0], in_img.shape[1]    # Размер исходного изображения
    new_H, new_W = out_img.shape[0], out_img.shape[1]   # Размер выходного изображения
    # Вычисляем значения производных
    f_x, f_y, f_xy = derivative_calculation(in_img)
    # Обход пикселей выходного изображения
    for x in range(new_W):
        # Координата на исходном изображении по ширине
        w = x / Xfactor
        # Обработка границ
        if w < 0: w = 0
        if w >= width - 1: w = width - 1
        # Ближайшие узлы по ширине
        floor_w = int(np.floor(w))
        ceil_w = int(np.ceil(w))
        # Относительное значение координаты
        relative_w = w - floor_w
        for y in range(new_H):
            # Координата на исходном изображении по высоте
            h = y / Yfactor
            # Обработка границ
            if h < 0: h = 0
            if h >= height - 1: h = height - 1
            # Ближайшие узлы по высоте
            floor_h = int(np.floor(h))
            ceil_h = int(np.ceil(h))
            # Относительное значение координаты
            relative_h = h - floor_h
            # Создаем вектор-столбец свободных членов
            beta = np.array([[in_img[floor_h, floor_w]], [in_img[ceil_h, floor_w]], [in_img[floor_h, ceil_w]], [in_img[ceil_h, ceil_w]],
                             [f_y[floor_h, floor_w]], [f_y[ceil_h, floor_w]], [f_y[floor_h, ceil_w]], [f_y[ceil_h, ceil_w]],
                             [f_x[floor_h, floor_w]], [f_x[ceil_h, floor_w]], [f_x[floor_h, ceil_w]], [f_x[ceil_h, ceil_w]],
                             [f_xy[floor_h, floor_w]], [f_xy[ceil_h, floor_w]], [f_xy[floor_h, ceil_w]], [f_xy[ceil_h, ceil_w]]], 
                            dtype = np.double)
            # Вычисляем значения коэффициентов a_i
            alpha = np.matmul(invH, beta)
            # Вычисляем яркость
            p_xy = 0
            for i in range(4):
                for j in range(4):
                    p_xy += alpha[4*i + j] * (relative_w**i) * (relative_h**j)
            if p_xy > 255:
                out_img[y, x] = 255
            elif p_xy < 0:
                out_img[y, x] = 0
            else:
                out_img[y, x] = p_xy
    out_img = out_img.astype(np.uint8)
    return out_img



# Обработка одного цветового канала (поворот)
def bicubic_one_channel_rotate(in_img, out_img, cos_rad, sin_rad, x_left, y_top):
    height, width = in_img.shape[0], in_img.shape[1]    # Размер исходного изображения
    new_H, new_W = out_img.shape[0], out_img.shape[1]   # Размер выходного изображения
    # Вычисляем значения производных
    f_x, f_y, f_xy = derivative_calculation(in_img)
    # Центр исходного изображения
    h0 = height / 2
    w0 = width / 2
    # Обход пикселей выходного изображения
    for x in range(new_W):
        # Смещение координат по ширине
        delta_x = x + x_left - w0
        # Константы относительно цикла по высоте
        delta_x_cos = delta_x * cos_rad
        delta_x_sin = delta_x * sin_rad
        for y in range(new_H):
             # Смещение координат по высоте
            delta_y = y + y_top - h0
            # Координаты на исходном изображении
            h = delta_y * cos_rad - delta_x_sin + h0
            w = delta_y * sin_rad + delta_x_cos + w0
            # Обработка границ
            if -1 <= h < 0: h = 0
            if height - 1 <= h <= height + 1: h = height - 1
            if -1 <= w < 0: w = 0
            if width - 1 <= w <= width + 1: w = width - 1
            # Если вышли за границы - заполняем фон
            if h >= height or w >= width or h < 0 or w < 0:
                out_img[y, x] = 255
                continue
            # Ближайшие к этой точке узлы
            floor_h = int(np.floor(h))
            ceil_h = int(np.ceil(h))
            floor_w = int(np.floor(w))
            ceil_w = int(np.ceil(w))
            # Относительные значения координат
            relative_h = h - floor_h
            relative_w = w - floor_w
            # Создаем вектор-столбец свободных членов
            beta = np.array([[in_img[floor_h, floor_w]], [in_img[ceil_h, floor_w]], [in_img[floor_h, ceil_w]], [in_img[ceil_h, ceil_w]],
                             [f_y[floor_h, floor_w]], [f_y[ceil_h, floor_w]], [f_y[floor_h, ceil_w]], [f_y[ceil_h, ceil_w]],
                             [f_x[floor_h, floor_w]], [f_x[ceil_h, floor_w]], [f_x[floor_h, ceil_w]], [f_x[ceil_h, ceil_w]],
                             [f_xy[floor_h, floor_w]], [f_xy[ceil_h, floor_w]], [f_xy[floor_h, ceil_w]], [f_xy[ceil_h, ceil_w]]], 
                            dtype = np.double)
            # Вычисляем значения коэффициентов a_i
            alpha = np.matmul(invH, beta)
            # Вычисляем яркость
            p_xy = 0
            for i in range(4):
                for j in range(4):
                    p_xy += alpha[4*i + j] * (relative_w**i) * (relative_h**j)
            if p_xy > 255:
                out_img[y, x] = 255
            elif p_xy < 0:
                out_img[y, x] = 0
            else:
                out_img[y, x] = p_xy
    out_img = out_img.astype(np.uint8)
    return out_img
            



# Бикубическая интерполяция 
def bicubic_interpolation(in_img, out_img, T, action):
    in_img = in_img.astype(np.float64)
    if in_img.ndim == 3:  # RGB
        temp = []
        if action == '2': # Масштабирование
            Xfactor, Yfactor = T[0, 0], T[1, 1]
            for i in range(0, 3):
                temp.append(bicubic_one_channel_scale(in_img[:,:,i], out_img, Xfactor, Yfactor))
        else:   # Поворот
            cos_rad, sin_rad = T[0, 0], T[0, 1]
            # Верхний левый угол исходного изображения на выходном
            # (вычислен заранее в функции rotate)
            x_left, y_top = out_img[0, 0], out_img[0, 1]
            for i in range(0, 3):
                temp.append(bicubic_one_channel_rotate(in_img[:,:,i], out_img, cos_rad, sin_rad, x_left, y_top))
        out_img = np.stack((temp[0], temp[1], temp[2]), axis=2)
        Image.fromarray((out_img),mode="RGB").show()

    else:  # Один канал
        if action == '2': # Масштабирование
            Xfactor, Yfactor = T[0, 0], T[1, 1]
            out_img = bicubic_one_channel_scale(in_img, out_img, Xfactor, Yfactor)
        else:   # Поворот
            cos_rad, sin_rad = T[0, 0], T[0, 1]
            # Верхний левый угол исходного изображения на выходном
            # (вычислен заранее в функции rotate)
            x_left, y_top = out_img[0, 0], out_img[0, 1]
            out_img = bicubic_one_channel_rotate(in_img, out_img, cos_rad, sin_rad, x_left, y_top)
        Image.fromarray((out_img),mode="L").show()

    return out_img

