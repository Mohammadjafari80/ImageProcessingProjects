import math
import cv2
import numpy as np

scaling_factor = 2
tracker = 0


class States(enumerate):
    near_left_eye = 1
    near_right_eye = 2
    far_left_eye = 3
    far_right_eye = 4


def onMouse_near(event, x, y, flags, param):
    global tracker
    if event == cv2.EVENT_LBUTTONDOWN:
        param.append((scaling_factor * x, scaling_factor * y))
        tracker += 1
        if tracker == States.near_right_eye:
            cv2.destroyWindow('img_near')


def onMouse_far(event, x, y, flags, param):
    global tracker
    if event == cv2.EVENT_LBUTTONDOWN:
        param.append((scaling_factor * x, scaling_factor * y))
        tracker += 1
        if tracker == States.far_right_eye:
            cv2.destroyWindow('img_far')


def normalize(img):
    return cv2.normalize(img, np.zeros(img.shape), 0, 255, cv2.NORM_MINMAX)


def find_scale_transition(points_s, points_d):
    A = np.empty((0, 3), float)

    x, y = points_s[0]
    A = np.concatenate((A,
                        np.array([[x, 1, 0]])),
                        axis=0)
    A = np.concatenate((A,
                        np.array([[y, 0, 1]])),
                        axis=0)

    x, y = points_s[1]
    A = np.concatenate((A,
                        np.array([[x, 1, 0]])),
                       axis=0)

    answer = np.linalg.solve(A, np.array(points_d).flatten()[:3])
    answer = np.array([answer[0], 0, answer[1], 0, answer[0], answer[2]])
    return np.hstack((answer, 0, 0, 1))


def backward_mapping(image_s, H_matrix):
    height, width, channels = image_s.shape
    transformed_s = np.zeros(shape=image_s.shape)
    H_matrix = np.linalg.inv(H_matrix)
    for i in range(height):
        for j in range(width):
            for k in range(channels):
                x, y, w = H_matrix @ np.array([i, j, 1])
                if 0 < x < height - 1 and 0 < y < width - 1:
                    chopped_x, chopped_y = math.floor(x), math.floor(y)
                    a, b = x - chopped_x, y - chopped_y
                    interpolation = \
                        (1 - b) * (1 - a) * image_s[chopped_x, chopped_y, k] + \
                        a * (1 - b) * image_s[chopped_x + 1, chopped_y, k] + \
                        b * (1 - a) * image_s[chopped_x, chopped_y + 1, k] + \
                        a * b * image_s[chopped_x + 1, chopped_y + 1, k]

                    transformed_s[i, j][k] = int(interpolation)

    return transformed_s


def distance(x, y):
    return np.sqrt((x[0]-y[0]) ** 2 + (x[1] - y[1]) ** 2)


def gaussian_low_pass(image_shape, s):
    mask = np.zeros(image_shape)
    rows, cols = image_shape
    center = (rows/2, cols/2)
    for x in range(cols):
        for y in range(rows):
            mask[y, x] = np.exp(((-distance((y, x), center) ** 2)/(2 * (s ** 2))))
    return mask


def gaussian_high_pass(image_shape, r):
    mask = np.zeros(image_shape)
    rows, cols = image_shape
    center = (rows/2, cols/2)
    for x in range(cols):
        for y in range(rows):
            mask[y, x] = 1 - np.exp(((-distance((y, x), center) ** 2)/(2 * (r ** 2))))
    return mask


def calculate_shifted_fft(channels):
    b, g, r = channels

    b_fft = np.fft.fft2(b)
    shifted_b = np.fft.fftshift(b_fft)

    g_fft = np.fft.fft2(g)
    shifted_g = np.fft.fftshift(g_fft)

    r_fft = np.fft.fft2(r)
    shifted_r = np.fft.fftshift(r_fft)

    return shifted_b, shifted_g, shifted_r


def get_mask_filters(img_shape, S, R):
    return gaussian_high_pass(img_shape, r=R), gaussian_low_pass(img_shape, s=S)


def get_masked_FFT(mask, shifted_fft):
    return mask * shifted_fft


def get_all_masked_FFTs(mask, channels_shifted_fft):
    return get_masked_FFT(mask, channels_shifted_fft[0]),\
           get_masked_FFT(mask, channels_shifted_fft[1]),\
           get_masked_FFT(mask, channels_shifted_fft[2])


def combine_masked_FFTs(hp_channels, lp_channels, alpha, beta):
    return (beta * lp_channels[0] + alpha * hp_channels[0]),\
           (beta * lp_channels[1] + alpha * hp_channels[1]),\
           (beta * lp_channels[2] + alpha * hp_channels[2])


def get_inverse_FFT(shifted_FFT):
    image_ishifted = np.fft.ifftshift(shifted_FFT)
    image = np.fft.ifft2(image_ishifted )
    return np.real(image)


def get_all_inverse_FFTs(shifted_channels):
    return get_inverse_FFT(shifted_channels[0]),\
           get_inverse_FFT(shifted_channels[1]),\
           get_inverse_FFT(shifted_channels[2])


def get_logarithm_magnitude(shifted_FFT):
    amplitude_image = np.abs(shifted_FFT)
    log_amplitude_image = np.log(1 + amplitude_image)
    return log_amplitude_image.astype(np.uint8)


def get_all_channels_logarithm_magnitude(shifted_channels_FTT):
    return get_logarithm_magnitude(shifted_channels_FTT[0]),\
           get_logarithm_magnitude(shifted_channels_FTT[1]),\
           get_logarithm_magnitude(shifted_channels_FTT[2])


def normalize(img):
    return cv2.normalize(img, np.zeros(img.shape), 0, 255, cv2.NORM_MINMAX)


points_near = []
img_near = cv2.imread('res19-near.jpg')
cv2.namedWindow('img_near')
cv2.setMouseCallback('img_near', onMouse_near, param=points_near)
img = cv2.resize(img_near, (int(img_near.shape[1] / scaling_factor),
                            int(img_near.shape[0] / scaling_factor)))
cv2.imshow('img_near', img)
cv2.waitKey(0)

points_far = []
img_far = cv2.imread('res20-far.jpg')
cv2.namedWindow('img_far')
cv2.setMouseCallback('img_far', onMouse_far, param=points_far)
img = cv2.resize(img_far, (int(img_far.shape[1] / scaling_factor),
                           int(img_far.shape[0] / scaling_factor)))
cv2.imshow('img_far', img)
cv2.waitKey(0)

H = find_scale_transition(points_near, points_far).reshape((3, 3))

transformed_near = backward_mapping(img_near, H)
transformed_near = transformed_near.astype(np.uint8)
cv2.imwrite('res21-near.jpg', transformed_near)
cv2.imwrite('res22-far.jpg', img_far)

img_near = transformed_near

b_near, g_near, r_near = cv2.split(img_near)
b_far, g_far, r_far = cv2.split(img_far)


shifted_b_near, shifted_g_near, shifted_r_near = calculate_shifted_fft((b_near, g_near, r_near))
shifted_b_far, shifted_g_far, shifted_r_far = calculate_shifted_fft((b_far, g_far, r_far))


log_magnitude_b_near, log_magnitude_g_near, log_magnitude_r_near = \
    get_all_channels_logarithm_magnitude((shifted_b_near, shifted_g_near, shifted_r_near))
dft_near = cv2.merge([log_magnitude_b_near, log_magnitude_g_near, log_magnitude_r_near])
cv2.imwrite('res23-dft-near.jpg', normalize(dft_near))


log_magnitude_b_far, log_magnitude_g_far, log_magnitude_r_far = \
    get_all_channels_logarithm_magnitude((shifted_b_far, shifted_g_far, shifted_r_far))
dft_far = cv2.merge([log_magnitude_b_far, log_magnitude_g_far, log_magnitude_r_far])
cv2.imwrite('res24-dft-far.jpg', normalize(dft_far))


r = 35
s = 15
hp_mask, lp_mask = get_mask_filters(b_near.shape, s, r)
cv2.imwrite(f'res25-highpass-{r}.jpg', hp_mask * 255)
cv2.imwrite(f'res26-lowpass-{s}.jpg', lp_mask * 255)


b_masked_near, g_masked_near, r_masked_near = get_all_masked_FFTs(hp_mask,
                                                                  (shifted_b_near, shifted_g_near, shifted_r_near))
b_high_passed, g_high_passed, r_high_passed = get_all_inverse_FFTs((b_masked_near, g_masked_near, r_masked_near))
image_high_pass = cv2.merge(get_all_channels_logarithm_magnitude((b_masked_near, g_masked_near, r_masked_near)))
cv2.imwrite('res27-highpassed.jpg', normalize(image_high_pass))


b_masked_far, g_masked_far, r_masked_far = get_all_masked_FFTs(lp_mask,
                                                               (shifted_b_far, shifted_g_far, shifted_r_far))
b_low_passed, g_low_passed, r_low_passed = get_all_inverse_FFTs((b_masked_far, g_masked_far, r_masked_far))
image_low_pass = cv2.merge(get_all_channels_logarithm_magnitude((b_masked_far, g_masked_far, r_masked_far)))
cv2.imwrite('res28-lowpassed.jpg', normalize(image_low_pass))


a, b = 1.3, 1
b_combined, g_combined, r_combined = combine_masked_FFTs((b_masked_near, g_masked_near, r_masked_near),
                                                         (b_masked_far, g_masked_far, r_masked_far),
                                                         alpha=a, beta=b)

img_avg = cv2.merge(get_all_channels_logarithm_magnitude((b_combined, g_combined, r_combined)))
cv2.imwrite('res29-hybrid.jpg', normalize(img_avg))


b_hybrid, g_hybrid, r_hybrid = get_all_inverse_FFTs((b_combined, g_combined, r_combined))
image_hybrid_near = cv2.merge([b_hybrid, g_hybrid, r_hybrid])

cv2.imwrite('res30-hybrid-near.jpg', image_hybrid_near)

image_hybrid_far = cv2.resize(image_hybrid_near, (int(image_hybrid_near.shape[1]/8), int(image_hybrid_near.shape[0]/8)))
cv2.imwrite('res31-hybrid-far.jpg', image_hybrid_far)
