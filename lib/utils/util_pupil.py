import numpy as np
import cv2


def distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))

def get_pupil(img, lms):
    height, width, _ = img.shape
    x = np.arange(width)
    y = np.arange(height)
    x_grid, y_grid = np.meshgrid(x, y)

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = np.zeros(img_gray.shape, np.uint8)
    lms_this = lms.astype(np.int64)

    # right eye
    center_eye_l = lms_this[36]
    center_eye_r = lms_this[39]
    center_eye_u = lms_this[37] / 2 + lms_this[38] / 2
    center_eye_d = lms_this[40] / 2 + lms_this[41] / 2
    center_eye = (center_eye_l + center_eye_r + center_eye_u + center_eye_d) / 4
    dis1_l = distance(center_eye_l, center_eye_r)
    dis2_l = distance(center_eye_u, center_eye_d)

    ptsr = lms_this[36:42]
    mask_r = cv2.polylines(mask.copy(), [ptsr], True, 1)
    mask_r = cv2.fillPoly(mask_r, [ptsr], 1)
    img_eye_r = img_gray * mask_r + (1 - mask_r) * 255
    thres = int(np.min(img_eye_r)) + 30
    mask_r = mask_r.astype(np.float32) * (img_eye_r < thres).astype(np.float32)

    if (np.sum(mask_r) < 10) or (dis2_l / dis1_l < 0.1):
        pupil_r = np.hstack((center_eye.copy(), np.array([0.0])))
    else:
        r_eye_x = np.sum(x_grid * mask_r) / np.sum(mask_r)
        r_eye_y = np.sum(y_grid * mask_r) / np.sum(mask_r)
        pupil_r = np.array([r_eye_x, r_eye_y, np.array(1.0)], dtype=np.float32)
    
    # left eye
    center_eye_l = lms_this[42]
    center_eye_r = lms_this[45]
    center_eye_u = lms_this[43] / 2 + lms_this[44] / 2
    center_eye_d = lms_this[46] / 2 + lms_this[47] / 2
    center_eye = (center_eye_l + center_eye_r + center_eye_u + center_eye_d) / 4
    dis1_l = distance(center_eye_l, center_eye_r)
    dis2_l = distance(center_eye_u, center_eye_d)

    ptsl = lms_this[42:48]#.reshape((-1, 1, 2))
    mask_l = cv2.polylines(mask.copy(), [ptsl], True, 1)
    mask_l = cv2.fillPoly(mask_l, [ptsl], 1)
    img_eye_l = img_gray * mask_l + (1 - mask_l) * 255
    thres = int(np.min(img_eye_l)) + 30
    mask_l = mask_l.astype(np.float32) * (img_eye_l < thres).astype(np.float32)

    if (np.sum(mask_l) < 10) or (dis2_l / dis1_l < 0.1):
        pupil_l = np.hstack((center_eye.copy(), np.array([0.0])))
    else:
        l_eye_x = np.sum(x_grid * mask_l) / np.sum(mask_l)
        l_eye_y = np.sum(y_grid * mask_l) / np.sum(mask_l)
        pupil_l = np.array([l_eye_x, l_eye_y, np.array(1.0)], dtype=np.float32)
    return np.vstack((pupil_r, pupil_l))