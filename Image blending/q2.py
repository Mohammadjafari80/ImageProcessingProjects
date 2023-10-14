# %%
import cv2
import numpy as np
import scipy
import scipy.sparse
from scipy.sparse import linalg as linalg
from scipy.sparse import lil_matrix as lil_matrix
from scipy.sparse.linalg import spsolve

# %%
def convert_bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def normalize(img):
    return cv2.normalize(img, np.zeros(img.shape), 0, 255, cv2.NORM_MINMAX)

# %%
k = 2 # Resizing Factor - The higher the rate - The Faster the Program
source = cv2.imread('res05.jpg')
height, width = source.shape[:2]
source = cv2.resize(source, (int(width/k), int(height/k)))
target = cv2.imread('res06.jpg')
target = cv2.resize(target, (int(width/k), int(height/k)))
mask = cv2.imread('mask-blend.jpg', cv2.IMREAD_GRAYSCALE)
mask = cv2.resize(mask, (int(width/k), int(height/k)))


# %%
mask_indices = np.where(mask > 0)
mask_indices = list(zip(*mask_indices))



moves = [(1, 0), (0, 1), (-1, 0), (0, -1)]

def apply_on_channel(source_channel, target_channel, L, mask_v, A, height, width):
    source_channel_vector = source_channel.flatten()
    target_channel_vector = target_channel.flatten()
    ratio = 0.75 # How Effective is source gradient.
    b = L @ (ratio * source_channel_vector + (1-ratio) * target_channel_vector)
    b[mask_v==0] = target_channel_vector[mask_v==0]
    x = spsolve(A, b)
    x = x.reshape((height, width))
    return clip(x)

def generate_mask(A, y_max, x_max):
    for y in range(1, y_max - 1):
        for x in range(1, x_max - 1):
            if mask[y, x] == 0:
                k = x + y * x_max
                A[k, k], A[k, k + 1], A[k, k - 1], A[k, k + x_max], A[k, k - x_max]  = 1, 0, 0, 0, 0

def get_laplacian(n, m):

    Diag = scipy.sparse.lil_matrix((m, m))
    Diag.setdiag(-1, -1)
    Diag.setdiag(4)
    Diag.setdiag(-1, 1)
        
    A = scipy.sparse.block_diag([Diag] * n).tolil()
    A.setdiag(-1, 1*m)
    A.setdiag(-1, -1*m)
    
    return A

def blend_poisson(source, target, mask):

    y_max, x_max = target.shape[:-1]

    mask = mask[:y_max, :x_max]    
    mask[mask != 0] = 1
    
    A = get_laplacian(y_max, x_max)

    L = A.tocsc()
    generate_mask(A, y_max, x_max)
    A = A.tocsc()

    mask_vector = mask.flatten()   

    for channel in range(3):
        target[:y_max, :x_max, channel] = \
        apply_on_channel(source[:y_max, :x_max, channel], target[:y_max, :x_max, channel], L, mask_vector, A, y_max, x_max)
        
    return target

def clip(v):
    new_v = np.copy(v)
    new_v[new_v > 255] = 255
    new_v[new_v < 0] = 0
    return new_v.astype(np.uint8)

# %%
result = blend_poisson(source, target, mask)

# %%
cv2.imwrite("res07.jpg", result)


