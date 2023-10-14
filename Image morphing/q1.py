import cv2
import numpy as np
from scipy.spatial import Delaunay
import imageio


POINT_SELECTION_MODE = False  # You can select points manually if you make it True
scaling_factor = 2.5
count = 0
img_1 = cv2.imread('res01.png')
img_2 = cv2.imread('res02.png')
height, width = img_1.shape[:2]
points_1 = [[0, 0], [0, width - 1], [height - 1, 0], [height - 1, width - 1]]
points_2 = [[0, 0], [0, width - 1], [height - 1, 0], [height - 1, width - 1]]


def sign(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1]);


def is_point_in_triangle(p, v1, v2, v3):
    d_1 = sign(p, v1, v3)
    d_2 = sign(p, v2, v3)
    d_3 = sign(p, v1, v2)

    has_neg = d_1 < 0 or d_2 < 0 or d_3 < 0
    has_pos = d_1 > 0 or d_2 > 0 or d_3 > 0

    return not (has_neg and has_pos)


def onMouse(event, x, y, flags, param):
    global count
    if event == cv2.EVENT_LBUTTONDOWN:
        count += 1
        if count % 2 == 1:
            cv2.rectangle(img_f, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), 2)
            points_1.append((int(scaling_factor * y), int(scaling_factor * x)))
            cv2.imshow('Select_Point', img_s)
        else:
            cv2.rectangle(img_s, (x - 5, y - 5), (x + 5, y + 5), (0, 255, 255), 2)
            points_2.append((int(scaling_factor * y), int(scaling_factor * x)))
            cv2.imshow('Select_Point', img_f)
    elif event == cv2.EVENT_RBUTTONDOWN:
        cv2.destroyWindow('Select_Point')


points = []

simplex = []
for i in range(height):
    for j in range(width):
        simplex.append((i, j))


def sign(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


def is_point_in_triangle(p, v1, v2, v3):
    d_1 = sign(p, v1, v3)
    d_2 = sign(p, v2, v3)
    d_3 = sign(p, v1, v2)
    has_neg = d_1 < 0 or d_2 < 0 or d_3 < 0
    has_pos = d_1 > 0 or d_2 > 0 or d_3 > 0

    return not (has_neg and has_pos)


def fast_convex_hull(new_points, tri, last_hull):
    new_hull = last_hull.copy()
    for i in range(height):
        print(i)
        for j in range(width):
            v1, v2, v3 = new_points[tri.simplices[last_hull[i][j]]]
            if is_point_in_triangle([i, j], v1, v2, v3):
                continue
            neighbours = tri.neighbors[last_hull[i][j]]
            neighbours = np.where(neighbours > 0, neighbours, np.max(neighbours))
            unique_neighbors = np.unique(neighbours)
            did_find = False
            for n in unique_neighbors:
                v1, v2, v3 = new_points[tri.simplices[n]]
                if is_point_in_triangle([i, j], v1, v2, v3):
                    new_hull[i][j] = n
                    did_find = True
                    break
            if did_find:
                continue
            for i in range(len(tri.simplices)):
                v1, v2, v3 = new_points[tri.simplices[i]]
                if is_point_in_triangle([i, j], v1, v2, v3):
                    new_hull[i][j] = i
                    break
    return new_hull


def backward_mapping(image_s, image_d, tri, transformations, new_points):
    rows, cols, channels = image_s.shape

    simplex_ind = tri.find_simplex(np.array(simplex)).reshape((rows, cols))
    segments = np.unique(simplex_ind)

    for segment in segments:
        k1, k2, k3 = tri.simplices[segment]
        H_matrix = transformations[(k1, k2, k3)]
        image_s[simplex_ind == segment] = cv2.warpAffine(image_d, H_matrix, (cols, rows), flags=2)[
            simplex_ind == segment]


if POINT_SELECTION_MODE:
    cv2.namedWindow('Select_Point')
    cv2.setMouseCallback('Select_Point', onMouse, param=points)
    img_f = cv2.resize(img_1, (int(img_1.shape[1] / scaling_factor),
                               int(img_1.shape[0] / scaling_factor)))
    img_s = cv2.resize(img_2, (int(img_2.shape[1] / scaling_factor),
                               int(img_2.shape[0] / scaling_factor)))
    cv2.imshow('Select_Point', img_f)
    cv2.waitKey(0)
else:
    file_lines = ''
    with open('./points_1.txt', 'r', encoding='UTF8') as f:
        file_lines = f.read().strip().split('\n')

    number_of_points = len(file_lines)
    points_1 = list()
    for i in range(number_of_points):
        x, y = map(float, file_lines[i].split())
        points_1.append((x, y))

    file_lines = ''
    with open('./points_2.txt', 'r', encoding='UTF8') as f:
        file_lines = f.read().strip().split('\n')

    number_of_points = len(file_lines)
    points_2 = list()
    for i in range(number_of_points):
        x, y = map(float, file_lines[i].split())
        points_2.append((x, y))

points_1, points_2 = np.array(points_1), np.array(points_2)
move_vector = points_2 - points_1
middle = (points_1 + points_2) / 2

tri = Delaunay(points_1)

steps = 45

first_image_sequence = [img_1]
T = np.linspace(0, 1, steps)
Tris = []

current_points = points_1
for i, t in enumerate(T):
    transforms = dict()
    new_points = points_1 + t * move_vector
    new_tri = Delaunay(new_points)
    Tris.append(new_tri)
    for triangle in new_tri.simplices:
        x, y, z = triangle
        before = current_points[triangle].astype(np.float32)
        after = new_points[triangle].astype(np.float32)
        before = np.roll(before, shift=1, axis=1)
        after = np.roll(after, shift=1, axis=1)
        transforms[(x, y, z)] = cv2.getAffineTransform(before, after)

    current_points = new_points.copy()
    result = np.zeros(shape=(height, width, 3), dtype=np.float64)
    backward_mapping(result, first_image_sequence[-1], new_tri, transforms, new_points)
    result = result.astype(np.uint8)
    first_image_sequence.append(result)

for i in range(len(first_image_sequence)):
    first_image_sequence[i] = cv2.cvtColor(first_image_sequence[i], cv2.COLOR_BGR2RGB)

second_image_sequence = [img_2]
T = np.linspace(0, 1, steps)

current_points = points_2
for i, t in enumerate(T):
    transforms = dict()
    new_points = points_2 - t * move_vector
    new_tri = Tris[-i - 1]
    for triangle in new_tri.simplices:
        x, y, z = triangle
        before = current_points[triangle].astype(np.float32)
        after = new_points[triangle].astype(np.float32)
        before = np.roll(before, shift=1, axis=1)
        after = np.roll(after, shift=1, axis=1)
        transforms[(x, y, z)] = cv2.getAffineTransform(before, after)

    current_points = new_points.copy()
    result = np.zeros(shape=(height, width, 3), dtype=np.float64)
    backward_mapping(result, second_image_sequence[-1], new_tri, transforms, new_points)
    result = result.astype(np.uint8)
    second_image_sequence.append(result)

for i in range(len(second_image_sequence)):
    second_image_sequence[i] = cv2.cvtColor(second_image_sequence[i], cv2.COLOR_BGR2RGB)

appended = []
for first, second, t in zip(first_image_sequence, second_image_sequence[::-1], T):
    appended.append(((1 - t) * first + t * second).astype(np.uint8))

appended += appended[::-1]

cv2.imwrite('res03.jpg', cv2.cvtColor(appended[14].astype(np.uint8), cv2.COLOR_RGB2BGR))
cv2.imwrite('res04.jpg', cv2.cvtColor(appended[29].astype(np.uint8), cv2.COLOR_RGB2BGR))

imageio.mimsave('./morph-3s.mp4', appended, fps=30)  # 30 fps
imageio.mimsave('./morph-6s.mp4', appended, fps=15)  # 6s
