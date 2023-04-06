from func import *


def task1():
    create_image(800, 800, 0, 'images/task_1a.jpg')
    create_image(800, 800, 255, 'images/task_1b.jpg')
    create_image(800, 800, [255, 0, 0], 'images/task_1c.jpg')
    create_image(800, 800, 'gradient', 'images/task_1d.jpg')


def task2():
    clr = Color([128, 0, 0])
    img = Img(200, 200)
    img.draw_star(img.line1, clr)
    img.save('images/task_2a.jpg')
    img.clear()

    img.draw_star(img.line2, clr)
    img.save('images/task_2b.jpg')
    img.clear()

    img.draw_star(img.line3, clr)
    img.save('images/task_2c.jpg')
    img.clear()

    img.draw_star(img.line4, clr)
    img.save('images/task_2d.jpg')
    img.clear()


def task5():
    img = Img(1000, 1000)
    mod = Model3D('model_1.obj')
    mod.screen_vertices(4000, 500)
    mod.draw_vertices(img, Color([255, 255, 255]))
    img.save_flipped('images/task_5.jpg')


def task7():
    img = Img(1000, 1000)
    mod = Model3D('model_1.obj')
    mod.screen_vertices(4000, 500)
    mod.draw_polygon(img, Color([255, 255, 255]))
    img.save_flipped('images/task_7.jpg')


def task10():
    img = Img(1000, 1000)
    mod = Model3D('model_1.obj')
    mod.screen_vertices(4000, 500)
    mod.draw_triangle(img, False)
    img.save_flipped('images/task_10.jpg')


def task11():
    img = Img(1000, 1000)
    mod = Model3D('model_1.obj')
    mod.screen_vertices(4000, 500)
    mod.draw_triangle(img, True)
    img.save_flipped('images/task_11.jpg')


def task12():
    img = Img(1000, 1000)
    mod = Model3D('model_1.obj')
    mod.screen_vertices(4000, 500)

    mod.draw_triangle(img, True)
    img.save_flipped('images/task_12.jpg')


def task15():
    img = Img(1000, 1000)
    mod = Model3D('model_1.obj')
    mod.screen_vertices(4000, 500)
    mod.draw_triangle(img, True)
    img.save_flipped('images/task_15.jpg')


def task18():
    img = Img(1000, 1000)
    mod = Model3D('model_1.obj')
    mod.projective_vertices()
    mod.draw_triangle(img, True)
    img.save('images/task18.jpg')


def dop():
    img = Img(1000, 1000)
    mod = Model3D('model_1.obj')
    mod.projective_vertices([0, -2 * np.pi / 3, 0])
    mod.draw_triangle(img, True)
    img.save('images/dop.jpg')


if __name__ == '__main__':
    # task1()
    # task2()
    # task5()
    # task7()
    # task10()
    # task11()
    # task12()
    # task15()
    # task18()
    dop()
    pass
