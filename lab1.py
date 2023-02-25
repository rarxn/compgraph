from PIL import Image
import numpy as np


def create_image(h, w, color, output_file):
    data = np.zeros((h, w, 3), dtype=np.uint8)
    if color == 'gradient':
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i, j] = (i + j) % 255
    else:
        data[:, :] = color
    img = Image.fromarray(data, 'RGB')
    img.save(output_file)


class Color:
    def __init__(self, rgb=[0, 0, 0]):
        self.rgb = rgb


class Img:
    def __init__(self, h=800, w=800, bg_color=Color([0, 0, 0])):
        self.h = h
        self.w = w
        self.data = np.zeros((h, w, 3), dtype=np.uint8)
        self.backgroundColor = bg_color

    def clear(self):
        self.data[:, :] = self.backgroundColor.rgb

    def save(self, filename):
        Image.fromarray(self.data, 'RGB').save(filename)

    def show(self):
        Image.fromarray(self.data, 'RGB').show()

    def setPixel(self, x, y, color):
        self.data[int(y), int(x)] = color.rgb

    def line1(self, x0, y0, x1, y1, color):
        t = 0.0
        while t < 1.:
            x = np.round(x0 * (1. - t) + x1 * t)
            y = np.round(y0 * (1. - t) + y1 * t)
            self.setPixel(x, y, color)
            t += 0.01

    def line2(self, x0, y0, x1, y1, color):
        for x in range(x0, x1 + 1):
            t = (x - x0) / (x1 - x0)
            y = np.round(y0 * (1. - t) + y1 * t)
            self.setPixel(x, y, color)

    def line3(self, x0, y0, x1, y1, color):
        steep = False
        if abs(x0 - x1) < abs(y0 - y1):
            x0, y0 = y0, x0
            x1, y1 = y1, x1
            steep = True
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        for x in range(x0, x1 + 1):
            t = (x - x0) / (x1 - x0)
            y = np.round(y0 * (1. - t) + y1 * t)
            self.setPixel(y, x, color) if steep else self.setPixel(x, y, color)

    def line4(self, x0, y0, x1, y1, color):
        steep = False
        if abs(x0 - x1) < abs(y0 - y1):
            x0, y0 = y0, x0
            x1, y1 = y1, x1
            steep = True
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        dx = x1 - x0
        dy = y1 - y0
        derror = abs(dy / dx)
        error = 0.
        y = y0
        for x in range(x0, x1 + 1):
            self.setPixel(y, x, color) if steep else self.setPixel(x, y, color)
            error += derror
            if error > 0.5:
                y += 1 if y1 > y0 else -1
                error -= 1.0
        pass

    def draw_star(self, type_lines, color):
        for i in range(13):
            alpha = 2 * np.pi * i / 13
            type_lines(100, 100, int(np.round(100 + 95 * np.cos(alpha))), int(np.round(100 + 95 * np.sin(alpha))),
                       color)


class Point3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class Model3D:
    def __init__(self, file_path):
        self.vertices = []
        self.poly = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.split()
                if line[0] == 'v':
                    self.vertices.append(Point3D(float(line[1]), float(line[2]), float(line[3])))
                elif line[0] == 'f':
                    self.poly.append([int(idx.split('/')[0]) - 1 for idx in line[1:]])

    def draw_vertices(self, image, color, k, b):
        for vertex in self.vertices:
            image.setPixel(k * vertex.x + b, -k * vertex.y + b, color)

    def draw_poly(self, image, color, k, b):
        for p in self.poly:
            for i in range(len(p)):
                p1 = self.vertices[p[i]]
                p2 = self.vertices[p[(i + 1) % len(p)]]
                image.line4((np.round(p1.x * k + b)).astype(int), (np.round(-p1.y * k + b)).astype(int),
                            (np.round(p2.x * k + b)).astype(int), (np.round(-p2.y * k + b)).astype(int), color)


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


def task4():
    img = Img(1000, 1000)
    mod = Model3D('model_1.obj')
    mod.draw_vertices(img, Color([255, 255, 255]), 4000, 500)
    img.save('images/task_4.jpg')


def taks6():
    img = Img(1000, 1000)
    mod = Model3D('model_1.obj')
    mod.draw_poly(img, Color([255, 255, 255]), 4000, 500)
    img.save('images/task_6.jpg')


if __name__ == '__main__':
    task1()
    task2()
    task4()
    taks6()
