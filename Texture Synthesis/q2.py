import numpy as np
import cv2


texture = cv2.imread('./tex-17.jpg')

box_size = 110
overlap = 30
final_texture_size = 2500


def minimum_cost_path(section_1, section_2):
    rows, cols = section_1.shape[:2]
    DP = np.zeros((rows, cols), dtype=np.float32)
    difference_matrix = np.sum(np.abs(section_1 - section_2), axis=2)
    DP[0, :] = difference_matrix[0, :]
    backtrace = np.zeros((rows, cols))  # 1:UP-LEFT  2:UP-CENTER 3:UP-RIGHT

    for row in range(rows):
        for col in range(cols):
            # Dynamic Programming
            possible_values = []

            if col == 0:
                possible_values = [1e5, DP[row - 1, col], DP[row - 1, col + 1]]
            elif col == cols - 1:
                possible_values = [DP[row - 1, col - 1], DP[row - 1, col], 1e5]
            else:
                possible_values = [DP[row - 1, col - 1], DP[row - 1, col], DP[row - 1, col + 1]]

            DP[row, col] = min(possible_values)
            backtrace[row, col] = possible_values.index(DP[row, col]) + 1
            DP[row, col] += difference_matrix[row, col]

    return DP, backtrace


def backtrace_path(DP, backtrace, vetical=True):
    rows, cols = DP.shape
    row_index = rows - 1
    col_index = np.argmin(DP[row_index])
    path = []
    while row_index >= 0:
        path.append((row_index, col_index) if vetical else (col_index, row_index))
        if backtrace[row_index, col_index] == 1:
            col_index -= 1
        elif backtrace[row_index, col_index] == 3:
            col_index += 1
        row_index -= 1

    return path[::-1]


def find_minimum_cost_path(section_1, section_2, vertical=True):
    DP, backtrace = minimum_cost_path(section_1, section_2) if vertical else \
        minimum_cost_path(cv2.transpose(section_1),
                          cv2.transpose(section_2))

    return backtrace_path(DP, backtrace, vertical)


def create_blend_masks_horizontal(path, overlap_shape):
    rows, cols = overlap_shape
    mask_left = np.zeros((rows, cols))
    mask_right = np.ones((rows, cols))
    for row, col in path:
        mask_left[row, :col] = np.linspace(1, 0.8, num=col, endpoint=False)
        mask_left[row, col] = 0.5
        mask_left[row, col + 1:] = np.linspace(0.2, 0, num=cols - col - 1, endpoint=True)

    return mask_left, mask_right - mask_left


def create_blend_masks_vertical(path, overlap_shape):
    rows, cols = overlap_shape
    mask_up = np.zeros((rows, cols))
    mask_down = np.ones((rows, cols))
    for row, col in path:
        mask_up[:row, col] = np.linspace(1, 0.8, num=row, endpoint=False)
        mask_up[row, col] = 0.5
        mask_up[row + 1:, col] = np.linspace(0.2, 0, num=rows - row - 1, endpoint=True)

    return mask_up, mask_down - mask_up


def handle_masks(current_row, current_col, large_texture, patch):
    path_v = []
    path_h = []
    mask_patch = np.ones((box_size, box_size))
    mask_large_texture = np.zeros((box_size, box_size))

    if current_row >= overlap:
        path_h = find_minimum_cost_path(
            large_texture[current_row: current_row + overlap, current_col: current_col + box_size],
            patch[:overlap, :],
            vertical=False)
    if current_col >= overlap:
        path_v = find_minimum_cost_path(
            large_texture[current_row: current_row + box_size, current_col: current_col + overlap],
            patch[:, :overlap],
            vertical=True)

    mask_left_right = create_blend_masks_horizontal(path_v, (box_size, overlap)) if current_col >= overlap else None
    mask_up_down = create_blend_masks_vertical(path_h, (overlap, box_size)) if current_row >= overlap else None

    if current_col >= overlap:
        mask_large_texture[:box_size, :overlap] = mask_left_right[0]
        mask_patch[:box_size, :overlap] = mask_left_right[1]

    if current_row >= overlap:
        mask_large_texture[:overlap, :box_size] = mask_up_down[0]
        mask_patch[:overlap, :box_size] = mask_up_down[1]

    return mask_patch, mask_large_texture


def normalize(img):
    return cv2.normalize(img, np.zeros(img.shape), 0, 255, cv2.NORM_MINMAX)


def clip_intensities(img):
    clip_img = np.copy(img)
    clip_img[clip_img > 255] = 255
    clip_img[clip_img < 0] = 0
    return clip_img


def find_random_best_match(reference_img, template, random_samples=5, vertical=True):
    ref_img = reference_img[:, :-box_size] if vertical else reference_img[:-box_size, :]

    res = cv2.matchTemplate(cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY),
                            cv2.cvtColor(template, cv2.COLOR_BGR2GRAY),
                            cv2.TM_CCOEFF_NORMED)
    possible_choices = np.argpartition((-res).flatten(), random_samples)
    random_choice = possible_choices[0]
    random_row, random_col = np.where(res == random_choice)[0]
    return random_row, random_col


def find_new_patch(current_row, current_col, large_texture, small_texture):
    mask = np.zeros((box_size, box_size, 3), dtype=np.uint8)
    template = np.zeros((box_size, box_size, 3), dtype=np.uint8)
    small_texture_gray = small_texture[:-int(box_size / 2), :-int(box_size / 2)]

    random_samples = 10

    if current_row > overlap:
        template[: overlap, :, :] = \
            large_texture[current_row: current_row + overlap, current_col: current_col + box_size, :]
        mask[: overlap, :, :] = 255

    if current_col > overlap:
        template[:, : + overlap, :] = \
            large_texture[current_row: current_row + box_size, current_col: current_col + overlap, :]
        mask[:, : overlap, :] = 255

    res = cv2.matchTemplate(small_texture_gray,
                            template,
                            cv2.TM_CCORR_NORMED,
                            mask=mask)

    ind = np.unravel_index(np.argsort(-res, axis=None), res.shape)
    random_choice = np.random.randint(random_samples)
    random_row, random_col = ind[0][random_choice], ind[1][random_choice]
    random_row, random_col = int(random_row + box_size / 2), int(random_col + box_size / 2)
    patch = small_texture[random_row: random_row + box_size, random_col: random_col + box_size]
    mask_patch, mask_large_texture = handle_masks(current_row, current_col, large_texture, patch)
    mask_p = np.zeros_like(patch, dtype=np.float32)
    mask_t = np.zeros_like(patch, dtype=np.float32)

    for i in range(3):
        mask_p[:, :, i] = mask_patch[:, :]
        mask_t[:, :, i] = mask_large_texture[:, :]

    image_patch = np.multiply(mask_p, patch, dtype=np.float32)
    image_texture = np.multiply(mask_t,
                                large_texture[current_row: current_row + box_size, current_col: current_col + box_size],
                                dtype=np.float32)

    return (image_patch + image_texture).astype(np.uint8)


def generate_texture(img):
    large_texture = np.zeros((final_texture_size + 200, final_texture_size + 200, 3), dtype=np.uint8)
    small_row, small_col = np.random.randint(min(img.shape[:2]) - box_size, size=2)
    large_row, large_col = 0, 0
    while large_row < final_texture_size:
        while large_col < final_texture_size:
            if large_row == 0 and large_col == 0:
                large_texture[:box_size, :box_size, :] = img[small_row: small_row + box_size,
                                                         small_col: small_col + box_size, :]
                large_col += (box_size - overlap)
                continue
            large_texture[large_row:large_row + box_size, large_col:large_col + box_size, :] = \
                find_new_patch(large_row, large_col, large_texture, img)

            large_col += (box_size - overlap)
        large_col = 0
        large_row += (box_size - overlap)

    return large_texture


generated_texture = generate_texture(texture)[:2500, :2500]
texture_small = np.zeros(shape=(2500, texture.shape[1] + 5, 3), dtype=np.uint8)
texture_small[:texture.shape[0], 5:] = texture
concatenated = cv2.hconcat([generated_texture, texture_small])
cv2.imwrite(f'res14.jpg', concatenated)

