import cv2
import numpy as np


def normalize(img):
    return cv2.normalize(img, np.zeros(img.shape), 0, 255, cv2.NORM_MINMAX)


def cross_correlation(image, template):
    return cv2.filter2D(image - np.mean(image), ddepth=-1, borderType=cv2.BORDER_REFLECT,
                        kernel=np.fliplr(np.flipud(template)) - np.mean(template))


img_rgb = cv2.imread('Greek-ship.jpg')
patch = cv2.imread('patch.png')[70:390, 80:120]
h, w = patch.shape[:2]
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

matching_result = cross_correlation(img_gray, patch_gray)
matching_result = normalize(matching_result)
cv2.imwrite('res.jpg', matching_result)

rectangle_patch = np.zeros((h , w - 15), dtype=np.uint8)
rectangle_patch[:, 5:-5] = 255

second_matching_result = cross_correlation(matching_result.astype(np.uint8), rectangle_patch)

cv2.imwrite('rsess.jpg', normalize(second_matching_result))


second_matching_result = normalize(second_matching_result).astype(np.uint8)
threshold = 170
loc = np.where(second_matching_result >= threshold)
possible_points = sorted(zip(*loc[::-1]))
points = [possible_points[0]]
x_filter_begin = points[0][0]
template_threshold = 20
index = 0

for x, y in possible_points:
    if x - x_filter_begin >= template_threshold:
        index += 1
        x_filter_begin = x
        points.append((x, y))

    if second_matching_result[y, x] > second_matching_result[points[index][1], points[index][0]]:
        points[index] = x, y

for x, y in points:
    x_start, y_start = int(x - w / 2), int(y - h / 2)
    x_end, y_end = x_start + w, y_start + h
    cv2.rectangle(img_rgb, (x_start, y_start), (x_end, y_end), (30, 50, 255), 3)

cv2.imwrite('res15.jpg', img_rgb)

