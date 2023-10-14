import math
import cv2
import numpy as np

scaling_factor = 1.5
tracker = 0


class States(enumerate):
    upper_left_corner = 1
    upper_right_corner = 2
    lower_right_corner = 3
    lower_left_corner = 4


def onMouse(event, x, y, flags, param):
    global tracker
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.rectangle(img, (x - 10, y - 10), (x + 10, y + 10), (0, 0, 255), 2)
        param.append((int(scaling_factor * y), int(scaling_factor * x)))
        tracker += 1
        cv2.imshow('books', img)
        if tracker == States.lower_left_corner:
            cv2.destroyWindow('books')


def distance(x, y):
    return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


def find_projection(points_s, points_d):
    A = np.empty((0, 8), dtype=np.float64)
    b = np.empty(0)
    number_of_data = len(points_s)
    for i in range(number_of_data):
        x, y = points_s[i]
        x_d, y_d = points_d[i]
        A = np.concatenate((A,
                            np.array([[x, y, 1, 0, 0, 0, - x * x_d, - y * x_d]])),
                           axis=0)
        A = np.concatenate((A,
                            np.array([[0, 0, 0, x, y, 1, - x * y_d, - y * y_d]])),
                           axis=0)
        b = np.concatenate((b, np.array([x_d, y_d])))

    answer = np.linalg.solve(A, b)
    return np.hstack((answer, 1)).reshape((3, 3))


def backward_mapping(image_s, image_d, H_matrix):
    rows, cols, channels = image_s.shape
    H_matrix = np.linalg.inv(H_matrix)

    for i in range(rows):
        for j in range(cols):
            for k in range(channels):
                x, y, w = H_matrix @ np.array([i, j, 1])
                x, y = x / w, y / w
                if 0 <= x < image_d.shape[0] - 1 and 0 <= y < image_d.shape[1] - 1:
                    chopped_x, chopped_y = math.floor(x), math.floor(y)
                    a, b = x - chopped_x, y - chopped_y
                    interpolation = np.array([[1 - a, a]]) @ \
                                     np.array([[image_d[chopped_x, chopped_y, k], image_d[chopped_x, chopped_y + 1, k]],
                                               [image_d[chopped_x + 1, chopped_y, k],
                                                image_d[chopped_x + 1, chopped_y + 1, k]]]) @ \
                                     np.array([[1 - b], [b]])
                    image_s[i, j, k] = interpolation


books_corner = []
books_img = cv2.imread('books.jpg')
cv2.namedWindow('books')
cv2.setMouseCallback('books', onMouse, param=books_corner)
img = cv2.resize(books_img, (int(books_img.shape[1] / scaling_factor),
                             int(books_img.shape[0] / scaling_factor)))
cv2.imshow('books', img)
cv2.waitKey(0)

p_1, p_2, p_3, p_4 = books_corner

width = int(distance(p_1, p_2) * 0.5 + distance(p_3, p_4) * 0.5) * 3
height = int(distance(p_2, p_3) * 0.5 + distance(p_4, p_1) * 0.5) * 3

homograph_corners = [(0, 0), (0, width), (height, width), (height, 0)]
H = find_projection(books_corner, homograph_corners)
projection_img = np.zeros(shape=(height, width, 3), dtype=np.float64)
backward_mapping(projection_img, books_img, H)
projection_img = projection_img.astype(np.uint8)
cv2.imwrite('res18.jpg', projection_img)

