import transformations as transform # Геометрические преобразования
import image as img # Работа с изображениями
import tests        # Создание тестовых картинок для отчета, вычисление psnr и ssim
import nearest_neighbor as nn   # Интерполяция по ближайшему соседу
import cubic_spline # Интерполяция кубическими сплайнами
import bicubic  # Бикубическая интерполяция
import bilinear # Билинейная интерполяция
import lanczos   # Интерполяция Ланцоша


def main():
    print("Имя файла: ", end = '')
    name = input()
    in_img = img.init_image(name)
    print("1 - Поворот, 2 - Масштабирование.\nВвод: ", end='')
    action = input()
    if action == '1' or action == '2':
        if action == '1':
            print("Значение угла задается в градусах.")
            print("При отрицательном значении угла - поворот по часовой стрелке, иначе - против.")
            print("angle = ", end = '')
            try:
                angle = float(input())
            except ValueError:
                return
            # Пустая матрица для выходного изображения и матрица преобразования
            out_img, T = transform.rotate(in_img, angle)
        else:
            print("Коэффициент масштабирования по ширине.")
            print("Xfactor = ", end = '')
            Xfactor = input()
            print("Коэффициент масштабирования по высоте.")
            print("Yfactor = ", end = '')
            Yfactor = input()
            try:
                Xfactor = float(Xfactor)
                Yfactor = float(Yfactor)
            except ValueError:
                return
            # Пустая матрица для выходного изображения и матрица преобразования
            out_img, T = transform.scaling(in_img, Xfactor, Yfactor)
    else:
        return
    print("Выбор метода интерполяции.")
    print("1 - Интерполяция по методу ближайшего соседа,\n"
          "2 - Билинейная интерполяция,\n"
          "3 - Интерполяция кубическими сплайнами,\n"
          "4 - Бикубическая интерполяция,\n"
          "5 - Интерполяция Ланцоша.\nВвод: ", end='')
    i = input()
    if i == '1':
        out_img = nn.nearest_neighbor_interpolation(in_img, out_img, T, action)
    elif i == '2':
        out_img = bilinear.bilinear_interpolation(in_img, out_img, T, action)
    elif i == '3':
        out_img = cubic_spline.spline_interpolation(in_img, out_img, T, action)
    elif i == '4':
        out_img = bicubic.bicubic_interpolation(in_img, out_img, T, action)
    elif i == '5':
        # Выбор параметра
        print("Необходимо задать размер области для интерполяции.\n"
              "При a = x для вычисления значения в неизвестной точке используется x^2 соседних пикселей.\n"
              "Допустимые значения - целые числа от 2 до 5. Стандартное значение: a = 3")
        print("Значение параметра a: a = ", end = '')
        a = input()
        try: 
            a = int(a)
        except ValueError:
            print("Недопустимое значение параметра.")
            return
        if a < 2 and a > 5:
            print("Недопустимое значение параметра.")
            return
        out_img = lanczos.lanczos_interpolation(in_img, out_img, T, action, a)
    else:
        return
    print("1 - Сохранить, 2 - Завершить.\nВвод: ", end = '')
    i = input()
    if i == '1':
        img.save_img(out_img)


main()


