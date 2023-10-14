import numpy as np
import cv2
import matplotlib.pylab as plt
from skimage import io
from skimage.feature import peak_local_max
import math

scaling_factor = 1
tracker = 0
image = cv2.imread('tasbih.jpg')
img = image.copy()
mask = np.zeros_like(image[:, :, 0])
initial_points = []
total_distance = 0


def onMouse(event, x, y, flags, param):
    global image
    global  total_distance
    if event == cv2.EVENT_LBUTTONDOWN:
        print(total_distance)
        if len(initial_points) != 0:
            image = cv2.line(image, initial_points[-1][::-1], (x, y), color=(0, 0, 255))
            total_distance += np.sqrt((initial_points[-1][0]-y)**2 + (initial_points[-1][1]-x)**2)
            cv2.imshow('contour', image)
        initial_points.append((y, x))
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        total_distance += np.sqrt((initial_points[-1][0] - initial_points[0][0]) ** 2 + (initial_points[-1][1] - initial_points[0][1]) ** 2)
        initial_points.append(initial_points[0])
        cv2.destroyWindow('contour')


cv2.namedWindow('contour')
cv2.setMouseCallback('contour', onMouse)
image = cv2.resize(image, (int(image.shape[1] / scaling_factor),
                           int(image.shape[0] / scaling_factor)))
cv2.imshow('contour', image)
cv2.waitKey(0)

def find_next_points(points):
    next = list()
    for point in points:
        next_points = list()
        for choice in choices.values():
            next_points.append(choice(*point))
        next.append(next_points)
    return np.array(next)


def calculate_average_distance(points):
    new_points = points.copy()
    new_points.append(points[0])
    p = np.array(new_points)
    x = p[:, 0]
    y = p[:, 1]
    x_distances = np.convolve(x, [1, -1], 'valid')
    y_distances = np.convolve(y, [1, -1], 'valid')
    total_distances = np.sum((x_distances ** 2 + y_distances ** 2) ** 0.5)
    return total_distances / (len(points))


def calculate_pair_distances(points):
    new_points = points.copy()
    new_points.append(points[0])
    p = np.array(new_points)
    x = p[:, 0]
    y = p[:, 1]
    x_distances = np.convolve(x, [1, -1], 'valid')
    y_distances = np.convolve(y, [1, -1], 'valid')
    total_distances = (x_distances ** 2 + y_distances ** 2) ** 0.5
    return total_distances


def calculate_distance(p1, p2, d):
    return (np.sum((np.array(p1) - np.array(p2)) ** 2, axis=-1) ** 0.5 - d) ** 2


def calculate_second_order_distance(p1, p2, p3):
    return np.sum((np.array(p1) - 2 * np.array(p2) + np.array(p3)), axis=-1) ** 2


def viterbi(source, points, current_points, G, d, alpha=0.1, beta=0.05, gama=0.1):
    points = np.array(points)[1:]
    score = np.zeros(shape=(points.shape[0] + 1, 9), dtype=np.float64)
    path = np.zeros(shape=(points.shape[0] + 1, 9))
    cols = points.shape[0]
    center = np.mean(np.array(current_points), axis=0)

    vectorized_distances = ((current_points - np.array(center)) ** 2).reshape(-1, 2)

    avg_distance = np.mean(np.sum(vectorized_distances, axis=-1) ** 0.5, )

    mean = center
    cov = [[d, 0], [0,d]]
    center = np.random.multivariate_normal(mean, cov, 1)
    theta = 1 / avg_distance
    c = 1

    for i in range(9):
        score[0, i] = alpha * calculate_distance(source, points[0, i], d) \
                      - gama * G[points[0, i][0], points[0, i][1]] \
                      + theta * beta * np.sum((np.array(points[0, i]) - center) ** 2) \
                      + c * calculate_second_order_distance(points[-1, 4], source, points[0, i])

    for j in range(9):
        new_scores = np.zeros(shape=(9,))
        for k in range(9):
            new_scores[k] = alpha * calculate_distance(points[0, k], points[1, j], d) + score[
                0, k] + c * calculate_second_order_distance(source, points[0, k], points[1, j])

        min_score = np.min(new_scores)
        min_index = np.argmin(new_scores)
        score[1, j] = min_score - gama * G[points[1, j][0], points[1, j][1]] + theta * beta * np.sum(
            (np.array(points[1, j]) - center) ** 2)
        path[1, j] = min_index

    for i in range(2, cols):
        for j in range(9):
            new_scores = np.zeros(shape=(9,))
            for k in range(9):
                new_scores[k] = alpha * calculate_distance(points[i - 1, k], points[i, j], d) + score[
                    i - 1, k] + c * calculate_second_order_distance(points[i - 2, int(path[i - 1, k])],
                                                                    points[i - 1, k], points[i, j])

            min_score = np.min(new_scores)
            min_index = np.argmin(new_scores)
            score[i, j] = min_score - theta * gama * G[points[i, j][0], points[i, j][1]] + theta * beta * np.sum(
                (np.array(points[i, j]) - center) ** 2)
            path[i, j] = min_index

    new_scores = np.zeros(shape=(9,))
    for i in range(9):
        new_scores[i] = alpha * calculate_distance(points[-1, i], source, d) + score[
            -2, i] + c * calculate_second_order_distance(points[-2, int(path[-2, i])], points[-1, i], source)
    min_score = np.min(new_scores)
    min_index = np.argmin(new_scores)

    for i in range(9):
        score[-1, i] = min_score - theta * gama * G[source[0], source[1]] + theta * beta * np.sum(
            (np.array(source) - center) ** 2)
        path[-1, i] = min_index

    return score, path


def backtrace_path(path_matrix):
    path = list()
    previous_point = path_matrix[-1, 0]
    path.append(previous_point)
    for i in range(path_matrix.shape[0] - 2, 0, -1):
        previous_point = path_matrix[i, int(previous_point)]
        path.append(previous_point)
    return path


def find_internal_paths(points, next, G, d, alpha, beta, gama):
    scores = np.zeros(shape=(9,), dtype=np.float64)
    paths = {}
    for i in range(9):
        score, path = viterbi(next[0, i], next, points, G, d, alpha, beta, gama)
        scores[i] = score[-1, 0]
        path = backtrace_path(path)
        path.append(i)
        paths[i + 1] = path[::-1]
    return scores, paths


def update_contour(path, points):
    new_points = list()
    for i, p in enumerate(path):
        new_points.append(choices[p + 1](*points[i]))
    return new_points


def minimize_energy(points, next, G, d, alpha=0.5, beta=1, gama=0.1):
    e_internal, p = find_internal_paths(points, next, G, d, alpha, beta, gama)
    e_total = e_internal
    ind = np.argmin(e_total)
    return update_contour(p[ind + 1], points)


ddepth = cv2.CV_32F
dx = cv2.Sobel(img, ddepth, 1, 0)
dy = cv2.Sobel(img, ddepth, 0, 1)
mag = (dx ** 2 + dy ** 2) ** 0.5
mag = cv2.cvtColor(mag, cv2.COLOR_BGR2GRAY)
mag = cv2.blur(mag, (11, 11))
mag[mag < 60] = 0
mag = cv2.medianBlur(mag, 5)
mag = cv2.medianBlur(mag, 5)

choices = {
    1: lambda x, y: (x - 1, y - 1),
    2: lambda x, y: (x - 1, y),
    3: lambda x, y: (x - 1, y + 1),
    4: lambda x, y: (x, y - 1),
    5: lambda x, y: (x, y),
    6: lambda x, y: (x, y + 1),
    7: lambda x, y: (x + 1, y - 1),
    8: lambda x, y: (x + 1, y),
    9: lambda x, y: (x + 1, y + 1)
}

k = 100
points = list()
for i in range(1, len(initial_points)):
    x1, y1 = initial_points[i-1]
    x2, y2 = initial_points[i]
    distance = np.sqrt((x1 - x2) ** 2 + (y1-y2) ** 2)
    ratio = distance / total_distance
    count = int(ratio * k)
    X = np.linspace(x1, x2, count).astype(np.uint16).tolist()
    Y = np.linspace(y1, y2, count).astype(np.uint16).tolist()
    for x, y in zip(X, Y):
        points.append((x, y))

print(len(points))
difference = float('inf')
# points_2 = points.copy()
points_2 = points.copy()
steps = []
while difference > 0:
    d = calculate_average_distance(points_2)
    next = find_next_points(points_2)
    new_points = minimize_energy(points_2, next, mag, d, alpha=1, beta=2, gama=1e3)
    difference = np.sum((np.array(new_points) - np.array(points_2)) ** 2)
    steps.append(new_points)
    points_2 = new_points
    # for index in zip(*np.where(np.abs(calculate_pair_distances(points_2)) > 2 * d)):
    #   points_2.insert((index[0]+1) % len(new_points), np.round((np.array(points_2[index[0]]) + np.array(points_2[(index[0]+1) % len(new_points)]))/2).astype(np.uint16).tolist())
    # new_points = points_2

import imageio

images = []

radius = 3
color = (255, 255, 0)

for step in steps:
    new_img = img.copy()
    for i in range(len(step)):
        new_img = cv2.line(new_img, tuple(step[i])[::-1], tuple(step[(i + 1) % len(step)])[::-1], color[::-1])

    for point in step:
        new_img = cv2.circle(new_img, tuple(point)[::-1], radius, color, thickness=-1)

    images.append(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))


images.append(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))

cv2.imwrite('res11.jpg', new_img)

imageio.mimsave('./contour.mp4', images) # modify duration