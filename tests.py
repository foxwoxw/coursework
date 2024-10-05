import matplotlib.pyplot as plt
import numpy as np
import skimage


# Построение графика двумерной функции яркости
# array - двумерный массив яркостей
def plot3D(array):
    b = np.arange(0, array.shape[1], 1) # координаты по ширине
    d = np.arange(0, array.shape[0], 1) # координаты по высоте
    # формирование двумерной сетки координат xy
    X, Y = np.meshgrid(b, d) 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.plot_surface(Y, X, array)
    plt.show()


# Дублирование соседних пикселей для тестовых картинок (один канал)
def duplicate_1(array, count):
    M, N = array.shape[0], array.shape[1]
    new_M = M * count
    new_N = N * count
    out_arr = np.zeros((new_M, new_N), dtype = np.uint8)
    h = 0
    w = 0
    for y in range(new_M):
        h = y // count
        for x in range(new_N):
            w = x // count
            out_arr[y, x] = array[h, w]
    return out_arr


# Дублирование соседних пикселей для тестовых картинок
def duplicate(in_img, count):
    if in_img.ndim == 3:  
        temp = []
        for i in range(0, 3):
            temp.append(duplicate_1(in_img[:,:,i], count))
        out_img = np.stack((temp[0], temp[1], temp[2]), axis=2) # объединяем каналы
    else:  
        out_img = duplicate_1(in_img, count)
    return out_img


# Среднеквадратичное отклонение
def calc_mse(img1, img2):
    mse = 0
    if img1.ndim == 3:
        for i in range(3):
            mse += np.mean((img1[:,:,i] - img2[:,:,i])**2)
        mse /= 3    # Усредняем
    else:
        mse = ((img1 - img2)**2)
    return mse


# Вычисление PSNR 
def calc_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = calc_mse(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


# Метрики сходства
def deviation(img1, img2):
    psnr = calc_psnr(img1, img2)
    ssim = skimage.metrics.structural_similarity(img1, img2, channel_axis=2)
    print("PSNR: ", psnr, sep = '')
    print("SSIM: ", ssim, sep = '')





