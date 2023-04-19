from PIL import Image
import numpy as np
from tqdm import tqdm

text_info = Image.open('test.jpg', 'r')
text_width, text_height = text_info.size
text_info = np.array(text_info)


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
    """Класс содержащий информацию о цвете в формате rgb"""

    def __init__(self, rgb=[0, 0, 0]):
        self.rgb = rgb


class Img:
    """класс для хранения изображения в памяти в виде массива"""

    def __init__(self, h=800, w=800, bg_color=Color([0, 0, 0])):
        self.h = h
        self.w = w
        self.data = np.zeros((h, w, 3), dtype=np.uint8)
        self.backgroundColor = bg_color
        self.z_buffer = np.zeros((h, w))

    def clear(self):
        self.data[:, :] = self.backgroundColor.rgb

    def save(self, filename):
        Image.fromarray(self.data, 'RGB').save(filename)

    def save_flipped(self, filename):
        img = Image.fromarray(self.data, 'RGB')
        img.transpose(Image.FLIP_TOP_BOTTOM).save(filename)

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

    def draw_star(self, type_lines, color):
        """Отрисовка звезды состоящей из линий"""
        for i in range(13):
            alpha = 2 * np.pi * i / 13
            type_lines(100, 100, int(np.round(100 + 95 * np.cos(alpha))), int(np.round(100 + 95 * np.sin(alpha))),
                       color)


class Point3D:
    """Класс точки с координатами x,y,z"""

    def __init__(self, x, y, z):
        self.coordinates = np.array([x, y, z])

    def get_coordinates(self):
        """Геттер для получения координат"""
        return self.coordinates

    def set_coordinates(self, coords):
        """Сеттер для установления новых координат"""
        self.coordinates = coords


class Model3D:
    def __init__(self, file_path):
        self.vertices = []
        self.polygon = []
        self.normal = []
        self.polygon_normal = []
        self.texture = []
        self.vt = []

        with open(file_path, 'r') as f:
            for line in f:
                if len(line) == 1:
                    continue
                line = line.split()
                if line[0] == 'v':
                    self.vertices.append(Point3D(float(line[1]), float(line[2]), float(line[3])))
                elif line[0] == 'f':
                    self.polygon.append([int(idx.split('/')[0]) - 1 for idx in line[1:]])
                    self.polygon_normal.append([int(idx.split('/')[2]) - 1 for idx in line[1:]])
                    self.texture.append([int(idx.split('/')[1]) - 1 for idx in line[1:]])
                elif line[0] == 'vn':
                    self.normal.append(Point3D(float(line[1]), float(line[2]), float(line[3])))
                elif line[0] == 'vt':
                    self.vt.append([float(line[1]), float(line[2])])

    def draw_vertices(self, image, color):
        """Отрисовка вершин трёхмерной модели"""
        for vertex in self.vertices:
            x, y, _ = vertex.get_coordinates()
            image.setPixel(x, y, color)

    def draw_polygon(self, image, color):
        """Отрисовка рёбер трёхмерной модели"""
        for p in self.polygon:
            for i in range(len(p)):
                x1, y1, _ = np.round(self.vertices[p[i]].get_coordinates()).astype(int)
                x2, y2, _ = np.round(self.vertices[p[(i + 1) % len(p)]].get_coordinates()).astype(int)
                image.line4(x1, y1, x2, y2, color)

    def draw_triangle(self, image, colored):
        """Отрисовка треугольников"""
        # colors = 3 if colored else 1
        l = [0, 0, 1]
        for p, n, t in tqdm(zip(self.polygon, self.polygon_normal, self.texture)):
            u0, v0 = self.vt[t[0]]
            u1, v1 = self.vt[t[1]]
            u2, v2 = self.vt[t[2]]
            u = np.array([u0, u1, u2])
            v = np.array([v0, v1, v2])
            # color = Color(np.random.randint(1, 256, size=colors))
            x0, y0, z0 = self.vertices[p[0]].get_coordinates()
            x1, y1, z1 = self.vertices[p[1]].get_coordinates()
            x2, y2, z2 = self.vertices[p[2]].get_coordinates()
            xmin = max(0, min(x0, x1, x2))
            ymin = max(0, min(y0, y1, y2))
            xmax = max(0, max(x0, x1, x2))
            ymax = max(0, max(y0, y1, y2))
            # task 18
            # vec_normal = np.array([np.dot(self.normal[idx].get_coordinates(), l) / (
            #         np.linalg.norm(self.normal[idx].get_coordinates()) * np.linalg.norm(l))
            #                        for idx in n])
            # task12-14
            # normal = get_normal(self.vertices[p[0]].get_coordinates(), self.vertices[p[1]].get_coordinates(),
            #                     self.vertices[p[2]].get_coordinates())
            # cos = np.dot(normal, l) / (np.linalg.norm(normal) * np.linalg.norm(l))
            # color = Color([255 * cos, 0, 0])
            # if cos < 0:  # task12-14
            for x in range(int(xmin), int(xmax + 1)):
                for y in range(int(ymin), int(ymax + 1)):
                    barycentric = get_barycentric_coordinates(x, y, x0, y0, x1, y1, x2, y2)
                    if all(b > 0 for b in barycentric):
                        z = barycentric[0] * z0 + barycentric[1] * z1 + barycentric[2] * z2
                        if 0 <= x < image.w and 0 <= y < image.h:
                            if z > image.z_buffer[x, y]:
                                idx0, idx1 = np.sum(text_width * (barycentric * u)), np.sum(
                                    text_height * (barycentric * v))
                                color = Color(text_info[int(np.round(idx0))][int(np.round(idx1))])
                                # color = Color([np.sum(255 * (vec_normal * barycentric)), 0, 0])  # task 18
                                image.z_buffer[x, y] = z
                                image.setPixel(x, y, color)

    def screen_vertices(self, k, b):
        """Преобразование к экранным координатам"""
        for v in self.vertices:
            v.set_coordinates(k * v.get_coordinates() + b)

    def projective_vertices(self, rotate=None):
        """Проективное преобразование"""
        R = rotation_matrix(*rotate) if rotate else None
        t = np.array([0.005, -0.05, 0.9])
        K = np.array([[-5000, 0, 500],
                      [0, -5000, 500],
                      [0, 0, 1]])
        for v in self.vertices:
            if R is not None:
                v.set_coordinates(R @ v.get_coordinates())
                v.set_coordinates(K @ (v.get_coordinates() + t))
            else:
                v.set_coordinates(K @ (v.get_coordinates() + t))
        if R is not None:
            for v in self.normal:
                v.set_coordinates(R @ v.get_coordinates())


def get_barycentric_coordinates(x, y, x0, y0, x1, y1, x2, y2):
    """
    вычисление барицентрических координат для точки с
    экранными (целочисленными координатами) (x,y) относительно
    вещественных вершин треугольника (x0, y0), (x1, y1) и (x2, y2)
    """
    lambda0 = ((x1 - x2) * (y - y2) - (y1 - y2) * (x - x2)) / \
              ((x1 - x2) * (y0 - y2) - (y1 - y2) * (x0 - x2))
    lambda1 = ((x2 - x0) * (y - y0) - (y2 - y0) * (x - x0)) / \
              ((x2 - x0) * (y1 - y0) - (y2 - y0) * (x1 - x0))
    lambda2 = ((x0 - x1) * (y - y1) - (y0 - y1) * (x - x1)) / \
              ((x0 - x1) * (y2 - y1) - (y0 - y1) * (x2 - x1))
    return lambda0, lambda1, lambda2


def get_normal(p0, p1, p2):
    return np.cross(p1 - p0, p2 - p0)


def rotation_matrix(alpha, betta, gamma):
    """Матрица для поворота модели"""
    r1 = np.array([[1, 0, 0],
                   [0, np.cos(alpha), np.sin(alpha)],
                   [0, -np.sin(alpha), np.cos(alpha)]])

    r2 = np.array([[np.cos(betta), 0, np.sin(betta)],
                   [0, 1, 0],
                   [-np.sin(betta), 0, np.cos(betta)]])

    r3 = np.array([[np.cos(gamma), np.sin(gamma), 0],
                   [-np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])
    return r1 @ r2 @ r3
