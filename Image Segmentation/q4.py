import math
import cv2
import numpy as np

scaling_factor = 4
BACKGROUND, FOREGROUND = 0, 1
tracker = 0
image = cv2.imread('birds.jpg')
img = cv2.resize(image, (int(image.shape[1] / scaling_factor),
                         int(image.shape[0] / scaling_factor)))
mask_label = 0
mask = np.ones_like(img[:, :, 0]) * 2
r_background = 20
r_foreground = 3


def onMouse(event, x, y, flags, param):
    global img, mask
    if event == cv2.EVENT_MOUSEMOVE:
        if mask_label == BACKGROUND:
            img = cv2.circle(img, (x, y), radius=r_background, color=(180, 100, 20), thickness=-1)
            mask = cv2.circle(mask, (x, y), radius=r_background, color=0, thickness=-1)
            cv2.imshow('Select Background', img)

    if event == cv2.EVENT_LBUTTONDOWN:
        if mask_label == FOREGROUND:
            img = cv2.circle(img, (x, y), radius=r_foreground, color=(0, 200, 120), thickness=-1)
            mask = cv2.circle(mask, (x, y), radius=r_foreground, color=1, thickness=-1)
            cv2.imshow('Select Foreground', img)

    if event == cv2.EVENT_LBUTTONDBLCLK:
        if mask_label == FOREGROUND:
            cv2.destroyWindow('Select Foreground')
        elif mask_label == BACKGROUND:
            cv2.destroyWindow('Select Background')


cv2.namedWindow('Select Background')
cv2.setMouseCallback('Select Background', onMouse)

cv2.imshow('Select Background', img)
cv2.waitKey(0)

mask_label += 1

cv2.namedWindow('Select Foreground')
cv2.setMouseCallback('Select Foreground', onMouse)

cv2.imshow('Select Foreground', img)
cv2.waitKey(0)

cv2.imwrite('mask.jpg', mask * 80)
# mask = cv2.imread('mask.jpg')


# mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
img = cv2.resize(image, (int(image.shape[1] / scaling_factor),
                         int(image.shape[0] / scaling_factor)))

mask = cv2.resize(mask, (int(image.shape[1] / scaling_factor),
                         int(image.shape[0] / scaling_factor)))


mask, bgdModel, fgdModel = cv2.grabCut(img, mask, None, bgdModel, fgdModel, 15, cv2.GC_INIT_WITH_MASK)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
mask2 = cv2.resize(mask2, (int(image.shape[1]), int(image.shape[0])))
mask_color = np.ones_like(image)
mask_color[:, :] = np.array([0, 0, 255])
mask_color = mask_color * mask2[:, :, np.newaxis]
alpha = 0.4
new_image = alpha * mask_color + (1 - alpha) * image
cv2.imwrite('res10.jpg', new_image)

