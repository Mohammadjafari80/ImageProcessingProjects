import cv2
import numpy as np


def remove_edges(image):
    indexes_w = image > 180
    indexes = indexes_w

    height, width = image.shape
    percision = 0.35

    begin_h = 0
    end_h = height
    begin_w = 0
    end_w = width

    limit = int(0.1 * width)

    for i in range(limit):
        begin_h += 1
        if np.count_nonzero(indexes[:, i]) < percision * height:
            break

    for i in range(limit):
        end_h -= 1
        if np.count_nonzero(indexes[:, width - i - 1]) < percision * height:
            break

    for i in range(limit):
        begin_w += 1
        if np.count_nonzero(indexes[i, :]) < percision * width:
            break

    for i in range(limit):
        end_w -= 1
        if np.count_nonzero(indexes[height - i - 1, :]) < percision * width:
            break

    return begin_h, end_h, begin_w, end_w


def remove_edges_by_sobel(image):
    height, width, dimension = image.shape
    gradient = apply_sobel(img=image)
    limit = int(max(0.06 * height, 0.06 * width))
    threshold = int(max(0.002 * height, 0.002 * width))
    indexes_w = gradient > 12
    precision = 0.60
    cv2.imwrite(f'gradient.jpg', gradient)
    cv2.imwrite(f'gradient_t.jpg', 255 * indexes_w)

    begin_height = 0
    end_height = height
    begin_width = 0
    end_width = width

    for i in range(limit):
        if np.count_nonzero(indexes_w[:, i]) > precision * height:
            begin_height = i + 1
        if np.count_nonzero(indexes_w[:, width - i - 1]) > precision * height:
            end_height = height - i - 1
        if np.count_nonzero(indexes_w[i, :]) > precision * width:
            begin_width = i + 1
        if np.count_nonzero(indexes_w[height - i - 1, :]) > precision * width:
            end_width = width - i - 1

    return begin_height + threshold, end_height - threshold, begin_width + threshold, end_width - threshold


def apply_sobel(img, kernel_size=51):
    height, width, dimension = img.shape
    resized_img = cv2.resize(src=img, dsize=(int(width * 1/4), int(height * 1/4)), interpolation=cv2.INTER_AREA)
    current_img = apply_gaussian_blur(resized_img, kernel_size)

    sobel_kernel_horizontal = np.array([[-1, 0, 1],
                                        [-2, 0, 2],
                                        [-1, 0, 1]])
    sobel_kernel_vertical = np.array([[-1, -2, -1],
                                      [0, 0, 0],
                                      [1, 2, 1]])

    B, G, R = cv2.split(current_img)

    G_final = np.maximum.reduce([calculate_gradient_magnitude(B, sobel_kernel_horizontal, sobel_kernel_vertical),
                                 calculate_gradient_magnitude(G, sobel_kernel_horizontal, sobel_kernel_vertical),
                                 calculate_gradient_magnitude(R, sobel_kernel_horizontal, sobel_kernel_vertical)])

    return cv2.resize(src=G_final, dsize=(width, height), interpolation=cv2.INTER_AREA)


def calculate_gradient_magnitude(channel, kernel_x, kernel_y):
    Gx = cv2.filter2D(src=channel, ddepth=-1, kernel=kernel_x)
    Gy = cv2.filter2D(src=channel, ddepth=-1, kernel=kernel_y)
    return np.sqrt(Gx ** 2 + Gy ** 2)


def apply_gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), cv2.BORDER_DEFAULT)


def resize_image(image, resize_factor: int):
    height, width = image.shape
    return cv2.resize(src=image, dsize=(int(width * resize_factor), int(height * resize_factor)),
                      interpolation=cv2.INTER_AREA)


def find_best_matches(channel_A, channel_B, channel_C, threshold: int, moved_positions=((0, 0), (0, 0))):
    return find_best_match(channel_A, channel_B, threshold, moved_positions[0]), \
           find_best_match(channel_A, channel_C, threshold, moved_positions[1])


def create_matched_channels(channel_A, channel_B, channel_C, threshold: int, optimal_position_set):
    height, width = channel_A.shape
    b, r = optimal_position_set
    b_x, b_y = b
    r_x, r_y = r

    b_channel = np.zeros(shape=(height + 2 * threshold, width + 2 * threshold))
    g_channel = np.zeros(shape=(height + 2 * threshold, width + 2 * threshold))
    r_channel = np.zeros(shape=(height + 2 * threshold, width + 2 * threshold))

    b_channel[threshold + b_x: threshold + b_x + height, threshold + b_y: threshold + b_y + width] = channel_A[:height,
                                                                                                     :width]
    g_channel[threshold: threshold + height, threshold: threshold + width] = channel_B[:height, :width]
    r_channel[threshold + r_x: threshold + r_x + height, threshold + r_y: threshold + r_y + width] = channel_C[:height,
                                                                                                     :width]

    return b_channel, g_channel, r_channel


def find_best_match(channel_A, channel_B, threshold_input: int, moved_positions=(0, 0)):
    height, width = channel_A.shape
    c_x, c_y = moved_positions
    threshold = max(abs(c_x) + threshold_input, abs(c_y) + threshold_input, threshold_input)

    score = np.zeros(shape=(2 * threshold_input + 1, 2 * threshold_input + 1), dtype=np.uint64)
    score = score - 1

    source_image = np.zeros(shape=(height + 2 * threshold, width + 2 * threshold))

    source_image[threshold: height + threshold, threshold: width + threshold] = channel_A[:height, :width]

    for i in range(-threshold_input, threshold_input + 1):
        for j in range(-threshold_input, threshold_input + 1):
            other_image = np.zeros(shape=(height + 2 * threshold, width + 2 * threshold))
            other_image[threshold + i + c_x: height + threshold + i + c_x,
            threshold + j + c_y: width + threshold + j + c_y] = channel_B[:height, :width]
            score[i + threshold_input, j + threshold_input] = calculate_score(source_image, other_image)

    i, j = np.unravel_index(score.argmin(), score.shape)

    return i - threshold_input + c_x, j - threshold_input + c_y


def calculate_score(src_img, other_img):
    return np.sum(np.abs(np.copy(src_img) - np.copy(other_img)))


# Load Image
original_image = cv2.imread('master-pnp-prok-01800-01886a.tif', 0)
original_image = original_image.astype(np.uint16)

# Remove white edges before processing
begin_h, end_h, begin_w, end_w = remove_edges(original_image)
original_image = original_image[begin_h: end_h, begin_w:end_w]

# Seperate BGR Channels
original_b_channel = original_image[0:int(original_image.shape[0] / 3), :]
original_g_channel = original_image[int(original_image.shape[0] / 3):int(original_image.shape[0] / 3) * 2, :]
original_r_channel = original_image[int(original_image.shape[0] / 3) * 2:int(original_image.shape[0] / 3) * 3, :]


# Resize Images to smaller scale
resize_constant = 1 / 8

resized_b_channel = resize_image(image=original_b_channel, resize_factor=resize_constant)
resized_g_channel = resize_image(image=original_g_channel, resize_factor=resize_constant)
resized_r_channel = resize_image(image=original_r_channel, resize_factor=resize_constant)


# Calculating the optimal position in bse image
optimal_positions = ()
threshold_const = 15
optimal_positions = find_best_matches(resized_g_channel,
                                      resized_b_channel,
                                      resized_r_channel,
                                      threshold_const)


while resize_constant < 1:

    optimal_positions = [x for optimal_set in optimal_positions for x in optimal_set]
    optimal_positions = [2 * x for x in optimal_positions]
    optimal_positions = tuple(tuple(optimal_positions[i:i + 2]) for i in range(0, 4, 2))
    resize_constant *= 2

    resized_b_channel = resize_image(image=original_b_channel, resize_factor=resize_constant)
    resized_g_channel = resize_image(image=original_g_channel, resize_factor=resize_constant)
    resized_r_channel = resize_image(image=original_r_channel, resize_factor=resize_constant)

    optimal_positions = find_best_matches(resized_g_channel,
                                          resized_b_channel,
                                          resized_r_channel,
                                          4,
                                          optimal_positions)

max_element = [abs(x) for optimal_set in optimal_positions for x in optimal_set]

optimized_b_channel, optimized_g_channel, optimized_r_channel = create_matched_channels(resized_b_channel,
                                                                                        resized_g_channel,
                                                                                        resized_r_channel,
                                                                                        max(max_element),
                                                                                        optimal_positions)
merged_img = cv2.merge([optimized_b_channel, optimized_g_channel, optimized_r_channel])
begin_h, end_h, begin_w, end_w = remove_edges_by_sobel(merged_img)
merged_img = merged_img[begin_h: end_h, begin_w:end_w]
cv2.imwrite('res03-sad.jpg', merged_img)
