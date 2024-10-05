import numpy as np
from PIL import Image


# Обработка одного цветового канала (масштабирование)
def bilinear_one_channel_scale(in_img, out_img, Xfactor, Yfactor):
    # Размер исходного изображения
    height, width = in_img.shape[0], in_img.shape[1]
    # Размер выходного изображения
    new_H, new_W = out_img.shape[0], out_img.shape[1]
    # Обход пикселей выходного изображения
    for y in range(new_H):
        # Исходные координаты по высоте
        h = y / Yfactor
        # Целые координаты
        h_floor = int(h)
        h_ceil = min(h_floor + 1, height - 1)
        for x in range(new_W):
            # Вычисляем исходные координаты по ширине
            w = x / Xfactor
            # Находим целые индексы и веса
            w_floor = int(w)
            w_ceil = min(w_floor + 1, width - 1)
             # Проверяем, чтобы индексы не выходили за пределы
            if 0 <= h_floor < height and 0 <= w_floor < width:
                # Находим веса
                weight_h = h - h_floor
                weight_w = w - w_floor
                # Билинейная интерполяция
                out_img[y, x] = (
                                (1 - weight_h) * (1 - weight_w) * in_img[h_floor, w_floor] +
                                weight_h * (1 - weight_w) * in_img[h_ceil, w_floor] +
                                (1 - weight_h) * weight_w * in_img[h_floor, w_ceil] +
                                weight_h * weight_w * in_img[h_ceil, w_ceil]
                                )
            else:
                out_img[y, x] = 255
    out_img = out_img.astype(np.uint8)
    return out_img



# Обработка одного цветового канала (поворот)
def bilinear_one_channel_rotate(in_img, out_img, cos_rad, sin_rad, x_left, y_top):
    # Размер исходного изображения
    height, width = in_img.shape[0], in_img.shape[1]
    # Размер выходного изображения
    new_H, new_W = out_img.shape[0], out_img.shape[1]
    # Центр исходного изображения
    h0 = height // 2
    w0 = width // 2
    # Обход пикселей выходного изображения
    for y in range(new_H):
        # Смещение координат по высоте
        delta_y = y + y_top - h0
        # Константы относительно цикла по ширине
        delta_y_cos = delta_y * cos_rad
        delta_y_sin = delta_y * sin_rad
        for x in range(new_W):
            # Смещение координат по ширине
            delta_x = x + x_left - w0
            # Обратное преобразование координат
            h = delta_y_cos - delta_x * sin_rad + h0
            w = delta_y_sin + delta_x * cos_rad + w0
            # Проверка выхода индексов за границы массива
            if 0 <= w < width and 0 <= h < height:
                w_floor, h_floor = int(np.floor(w)), int(np.floor(h))
                w_ceil, h_ceil = int(np.ceil(w)), int(np.ceil(h))
                # Проверка на выход за пределы взятых индексов
                if 0 <= w_floor < width and 0 <= h_floor < height and w_ceil < width and h_ceil < height:
                    # Билинейная интерполяция
                    coeff1 = (w_ceil - w) if w_ceil != w else 0
                    R1 = in_img[h_floor, w_floor] * coeff1 + in_img[h_floor, w_ceil] * (1 - coeff1)
                    coeff4 = (h - h_floor) if h_floor != h else 0
                    R2 = in_img[h_ceil, w_floor] * coeff1 + in_img[h_ceil, w_ceil] * (1 - coeff1)
                    # Заполнение нового массива
                    out_img[y, x] = (1 - coeff4) * R1 + coeff4 * R2
                else:
                    out_img[y, x] = 255
            else:
                out_img[y, x] = 255
    out_img = out_img.astype(np.uint8)
    return out_img




def bilinear_interpolation(in_img, out_img, T, action):    
    if in_img.ndim == 3:  # RGB
        temp = []
        if action == '2': # Масштабирование
            Xfactor, Yfactor = T[0, 0], T[1, 1]
            for i in range(0, 3):
                temp.append(bilinear_one_channel_scale(in_img[:,:,i], out_img, Xfactor, Yfactor))
        else:   # Поворот
            cos_rad, sin_rad = T[0, 0], T[0, 1]
            x_left, y_top = out_img[0, 0], out_img[0, 1]
            for i in range(0, 3):
                temp.append(bilinear_one_channel_rotate(in_img[:,:,i], out_img, cos_rad, sin_rad, x_left, y_top))
        out_img = np.stack((temp[0], temp[1], temp[2]), axis=2)
        Image.fromarray((out_img),mode="RGB").show()

    else:  # Один канал
        if action == '2': # Масштабирование
            Xfactor, Yfactor = T[0, 0], T[1, 1]
            out_img = bilinear_one_channel_scale(in_img, out_img, Xfactor, Yfactor)
        else:   # Поворот
            cos_rad, sin_rad = T[0, 0], T[0, 1]
            x_left, y_top = out_img[0, 0], out_img[0, 1]
            out_img = bilinear_one_channel_rotate(in_img, out_img, cos_rad, sin_rad, x_left, y_top)
        Image.fromarray((out_img),mode="L").show()

    return out_img