from PIL import Image
import numpy as np



# Интерполяция по ближайшему соседу для одного канала (масштабирование)
def nn_one_channel_scale(in_img, out_img, Xfactor, Yfactor):
    height, width = in_img.shape[0], in_img.shape[1]    # Размер входного изображения
    new_H, new_W = out_img.shape[0], out_img.shape[1]   # Размер выходного изображения
    for y in range(new_H):
        # Соответствующая координата по высоте на исх. изображении
        h = int(y / Yfactor)   
        # Проверяем не вышли ли за границы
        if h == height:
            h -= 1
        for x in range(new_W):
            # Соответствующая координата по ширине на исх. изображении
            w = int(x / Xfactor)
            # Проверяем не вышли ли за границы
            if w == width:
                w -= 1
            # Присваиваем значение яркости ближайшего пикселя
            out_img[y, x] = in_img[h, w]
    out_img = out_img.astype(np.uint8)
    return out_img


# Интерполяция по ближайшему соседу для одного канала (поворот)
def nn_one_channel_rotate(in_img, out_img, cos_rad, sin_rad, x_left, y_top):
    height, width = in_img.shape[0], in_img.shape[1]    # Размер входного изображения
    new_H, new_W = out_img.shape[0], out_img.shape[1]   # Размер выходного изображения
    # Координаты центра входного изображения
    h0 = height / 2
    w0 = width / 2
    # Обход пикселей выходного изображения
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
            # Координаты соответствующего пикселя на входном изображении
            h = int(np.floor(h))                
            w = int(np.floor(w))
            # Обработка граничных значений
            if h == height: h -= 1
            if w == width: w -= 1
            if h == -1: h += 1
            if w == -1: w += 1
            # Если вышли за границы - заполняем фон
            if h > height or w > width or h < 0 or w < 0:
                out_img[y, x] = 255
            # Иначе переписываем яркость ближайшего пикселя
            else:
                out_img[y, x] = in_img[h, w]
    out_img = out_img.astype(np.uint8)
    return out_img



# Интерполяция по ближайшему соседу
def nearest_neighbor_interpolation(in_img, out_img, T, action):
    in_img = in_img.astype(np.float64)
    if in_img.ndim == 3:  # RGB
        temp = []
        if action == '2': # Масштабирование
            Xfactor, Yfactor = T[0, 0], T[1, 1]
            for i in range(0, 3):
                temp.append(nn_one_channel_scale(in_img[:,:,i], out_img, Xfactor, Yfactor))
        
        else:   # Поворот
            cos_rad, sin_rad = T[0, 0], T[0, 1]
            # Верхний левый угол исходного изображения на выходном
            # (вычислен заранее в функции rotate)
            x_left, y_top = out_img[0, 0], out_img[0, 1]
            for i in range(0, 3):
                temp.append(nn_one_channel_rotate(in_img[:,:,i], out_img, cos_rad, sin_rad, x_left, y_top))
        out_img = np.stack((temp[0], temp[1], temp[2]), axis=2)
        Image.fromarray((out_img),mode="RGB").show()
    
    else:  # Один цветовой канал
        if action == '2': # Масштабирование
            Xfactor, Yfactor = T[0, 0], T[1, 1]
            out_img = nn_one_channel_scale(in_img, out_img, Xfactor, Yfactor)
        else:   # Поворот
            cos_rad, sin_rad = T[0, 0], T[0, 1]
            # Верхний левый угол исходного изображения на выходном
            # (вычислен заранее в функции rotate)
            x_left, y_top = out_img[0, 0], out_img[0, 1]
            out_img = nn_one_channel_rotate(in_img, out_img, cos_rad, sin_rad, x_left, y_top)
        Image.fromarray((out_img),mode="L").show()

    return out_img
    
