import numpy as np
import cv2

def create_mask(mask, offset):
  new_mask = mask.copy()
  gradient = np.linspace(1, 0, 2 * offset)
  center = int(mask.shape[1]/2) - 5
  new_mask[:, center - offset: center + offset] = gradient
  return new_mask

def blend(alpha, left, right):
  alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
  return alpha * left + (1- alpha) * right


img_left = cv2.imread('./res08.jpg')
img_left = cv2.resize(img_left, (int(img_left.shape[1]/3), int(img_left.shape[0]/3)))

img_right = cv2.imread('./res09.jpg')
img_right = cv2.resize(img_right, (int(img_right.shape[1]/3), int(img_right.shape[0]/3)))

masked = np.ones_like(img_right[:, :, 0], dtype=np.float32)
masked[:, int(img_right.shape[1]/2) - 5 :] = 0

kernel_sizes = [1, 7, 25, 45, 75, 199]
mask_offset_sizes = [15, 60, 120, 240, 400, 800]

mask_stack = [create_mask(masked, offset) for offset in mask_offset_sizes]

gaussian_stack_left = [cv2.blur(img_left, (k, k), borderType=cv2.BORDER_REFLECT) for k in kernel_sizes]
gaussian_stack_right = [cv2.blur(img_right, (k, k), borderType=cv2.BORDER_REFLECT) for k in kernel_sizes]

laplacian_stack_left = [np.subtract(gaussian_stack_left[i-1], gaussian_stack_left[i], dtype=np.float32) for i in range(1, 6)]
laplacian_stack_right = [np.subtract(gaussian_stack_right[i-1], gaussian_stack_right[i], dtype=np.float32) for i in range(1, 6)]

laplacian_stack_left.append(gaussian_stack_left[-1])
laplacian_stack_right.append(gaussian_stack_right[-1])

reconstruct_img = np.zeros_like(img_left, dtype=np.float32)
for i in range(1, len(laplacian_stack_left)+1):
  reconstruct_img += blend(mask_stack[-i], laplacian_stack_left[-i], laplacian_stack_right[-i])


cv2.imwrite('res10.jpg', reconstruct_img)

