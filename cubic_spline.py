import numpy as np
from PIL import Image


# Кубический сплайн S(x) на сетке x_i (i = 0..n; x_i < x_{i+1})
# S_i(x) = a_i + b_i*(x - x_i) + (c_i/2)*(x - x_i)^2 + (d_i/6)*(x - x_i)^3, x in [x_{i-1}, x_i]# 
class spline:
    def __init__(self, a, b, c, d, x):
        self.a = a  
        self.b = b  
        self.c = c  
        self.d = d  
        self.x = x  # Координата


# Построение сплайна
# x - узлы сетки; для всех узлов должно быть выполнено x_i < x_{i+1}
# y - значения функции f(x) (яркости) в узлах сетки
# n - количество узлов сетки (ширина или высота исходной картинки)
def build_spline(x, y, n):
    # Инициализация массива сплайнов (n сегментов => n многочленов)
    splines = [spline(0, 0, 0, 0, 0) for _ in range(0, n)]
    for i in range(0, n):
        splines[i].x = x[i]
        # Младший коэффициент сплайна равен значению функции f(x) в точке x_i
        splines[i].a = y[i]
    # Естественные граничные условия
    splines[0].c = splines[n - 1].c = 0.0
    # Решение СЛАУ относительно коэффициентов сплайнов c[i] методом прогонки 
    # Вычисление прогоночных коэффициентов alpha и beta - прямой ход метода прогонки
    n -= 1
    alpha = [0.0 for _ in range(0, n)]
    beta  = [0.0 for _ in range(0, n)]
    for i in range(1, n):
        # Длина участка, для которого x_i - правая граница; [x_{i-1}, x_i]
        h_i  = x[i] - x[i - 1]
        # Длина участка, для которого x_i - левая граница; [x_i, x_{i+1}]
        h_iplus1 = x[i + 1] - x[i]
        # Прогоночные коэффициенты
        k = 2 * (h_i + h_iplus1) + h_i * alpha[i - 1]
        alpha[i] = - h_iplus1 / k
        beta[i] = 1.0 / k * ((6 * ((y[i - 1] - y[i]) / h_iplus1 - (y[i] - y[i + 1]) / h_i)) - h_i * beta[i-1])
    # Нахождение решения - обратный ход метода прогонки
    n -= 1
    for i in range(n, 0, -1):
        splines[i].c = alpha[i] * splines[i + 1].c + beta[i]
    # По известным коэффициентам c[i] находим значения b[i] и d[i]
    n += 1
    for i in range(n, 0, -1):
        hi = x[i] - x[i - 1]
        splines[i].d = (splines[i].c - splines[i - 1].c) / h_i
        splines[i].b = h_i * (2.0 * splines[i].c + splines[i - 1].c) / 6.0 + (y[i] - y[i - 1]) / h_i
    return splines
 
    

# Вычисление яркости в точке
def interpolation(splines, x, n):
    # Бинарный поиск сплайна, приближающего значение яркости
    # на участке, которому принадлежит точка x
    i = 0
    j = n - 1
    while i + 1 < j:
        k = i + (j - i) // 2
        if x <= splines[k].x:
            j = k
        else:
            i = k
    s = splines[j] 
    # Относительное значение координаты
    dx = x - s.x
    # Вычисление значения найденного полинома в точке x
    p_xy = s.a + (s.b + (s.c / 2.0 + s.d * dx / 6.0) * dx) * dx
    if p_xy > 255: p_xy = 255
    elif p_xy < 0: p_xy = 0
    return p_xy


# Интерполяция кубическими сплайнами для одного канала (масштабирование)
def spline_one_channel_scale(in_img, out_img, Xfactor, Yfactor):
    height, width = in_img.shape[0], in_img.shape[1]    # Размерность матрицы исходного изображения    
    new_H, new_W = out_img.shape[0], out_img.shape[1]   # Размерность матрицы выходного изображения
    h_splines = []  # Список сплайнов для каждого из столбцов исходного изображения
    h_indices = [i for i in range(height)]  # Создаем сетку
    for i in range(width):
        h_splines.append(build_spline(h_indices, in_img[:,i], height))
    # Идем по пикселям выходного изображения вдоль столбцов
    # Параллельно составляем матрицу для последующего поиска сплайнов по строкам
    for_splines = np.zeros((new_H, width), dtype = np.double)
    for x in range(new_W):
        w = x / Xfactor     # Координата по ширине на входном изображении
        if w >= width: w = width - 1    # Обработка границ
        int_w = int(w)
        delta_w = w - int_w     # Относительное значение ширины
        for y in range(new_H):            
            h = y / Yfactor     # Координата по высоте на входном изображении            
            if h >= height: h = height - 1  # Обработка границ
            int_h = int(h)
            delta_h = h - int_h # Относительное значение высоты
            if delta_w == 0 and delta_h == 0:
                # Если координата целая (узел), то перезаписываем яркость из входного изображения
                out_img[y, x] = in_img[int_h, int_w]
            else:   # Иначе вычисляем яркость как значение соотв. сплайна в этой точке
                out_img[y, x] = interpolation(h_splines[int_w], h, height)
            for_splines[y, int_w] = out_img[y, x]
    # Находим сплайны для строк
    w_splines = []  # Список сплайнов для каждой строки
    w_indices = [i for i in range(width)]   # Создаем сетку
    for i in range(new_H):
        w_splines.append(build_spline(w_indices, for_splines[i], width))
    # Идем по пикселям выходного изображения вдоль строк
    for y in range(new_H):
        # Координата по высоте на входном изображении
        h = y / Yfactor
        # Обработка границ
        if h >= height: h = height - 1
        int_h = int(h)
        delta_h = h - int_h     # Относительное значение
        for x in range(new_W):
            # Находим соответствующую координату на входном изображении
            w = x / Xfactor   # Абсолютное значение
            # Обработка границ
            if w >= width:
                w = width - 1
            int_w = int(w)
            delta_w = w - int_w # Относительное значение
            # Если координата целая, значение уже известно, иначе считаем яркость 
            if delta_w != 0 or delta_h != 0:  
                out_img[y, x] = interpolation(w_splines[y], w, width)
    out_img = out_img.astype(np.uint8)
    return out_img



# Интерполяция кубическими сплайнами для одного канала (поворот)
# Только в одном направлении
def spline_one_channel_rotate(in_img, out_img, cos_rad, sin_rad, x_left, y_top):
    height, width = in_img.shape[0], in_img.shape[1]    # Размерность матрицы исходного изображения    
    new_H, new_W = out_img.shape[0], out_img.shape[1]   # Размерность матрицы выходного изображения
    # Находим сплайны для столбцов по входному изображению
    h_splines = []  # Список сплайнов для каждого столбца
    h_indices = [i for i in range(height)]  # Создаем сетку
    for i in range(width):
        h_splines.append(build_spline(h_indices, in_img[:,i], height))
    # Координаты центра входного изображения
    h0 = height / 2
    w0 = width / 2
    # Интерполяция по столбцам
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
            # Если вышли за границы - заполняем фон
            if h >= height or w >= width or h < 0 or w < 0:
                out_img[y, x] = 255
                continue
            # Относительные значения
            int_w = int(w)
            delta_w = w - int_w
            int_h = int(h)
            delta_h = h - int_h
            # Если узел - интерполяция не нужна
            if delta_h == 0 and delta_w == 0:
                out_img[y, x] = in_img[int_h, int_w]
            else:
                # Определяем яркость
                out_img[y, x] = interpolation(h_splines[int_w], h, height)
    out_img = out_img.astype(np.uint8)
    return out_img




# Интерполяция кубическими сплайнами (главная функция)
# Принимает матрицы входного(in_img) и выходного(out_img) изображений, матрицу прямого преобразования T
def spline_interpolation(in_img, out_img, T, action):
    in_img = in_img.astype(np.float64)
    if in_img.ndim == 3:  # RGB
        temp = []
        if action == '2': # Масштабирование
            Xfactor, Yfactor = T[0, 0], T[1, 1]
            for i in range(0, 3):
                temp.append(spline_one_channel_scale(in_img[:,:,i], out_img, Xfactor, Yfactor))
        else:   # Поворот
            cos_rad, sin_rad = T[0, 0], T[0, 1]
            # Верхний левый угол исходного изображения на выходном
            # (вычислен заранее в функции rotate)
            x_left, y_top = out_img[0, 0], out_img[0, 1]
            for i in range(0, 3):
                temp.append(spline_one_channel_rotate(in_img[:,:,i], out_img, cos_rad, sin_rad, x_left, y_top))
        out_img = np.stack((temp[0], temp[1], temp[2]), axis=2)
        Image.fromarray((out_img),mode="RGB").show()
    else:  # Один канал
        if action == '2': # Масштабирование
            Xfactor, Yfactor = T[0, 0], T[1, 1]
            out_img = spline_one_channel_scale(in_img, out_img, Xfactor, Yfactor)
        else:   # Поворот
            cos_rad, sin_rad = T[0, 0], T[0, 1]
            # Верхний левый угол исходного изображения на выходном
            # (вычислен заранее в функции rotate)
            x_left, y_top = out_img[0, 0], out_img[0, 1]
            out_img = spline_one_channel_rotate(in_img, out_img, cos_rad, sin_rad, x_left, y_top)
        Image.fromarray((out_img),mode="L").show()
    return out_img
