import numpy as np
import cv2
import math

from scipy.spatial.distance import cdist
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components
import torch
from torchvision.ops import nms


def binarize_img(cropImg: cv2.Mat):
    h, s, v = cv2.split(cv2.cvtColor(cropImg, cv2.COLOR_BGR2HSV))
    _, img_bin = cv2.threshold(s, 20, 255, cv2.THRESH_BINARY)
    return img_bin


def detect_chalk_text(cropImg: cv2.Mat, ocr, threshold=80):
    img_gray = cv2.cvtColor(cropImg, cv2.COLOR_BGR2GRAY)
    # img_bin = cv2.ximgproc.niBlackThreshold(img_gray, 255, cv2.THRESH_BINARY_INV, 31, 0.1, 
    #                                         binarizationMethod=cv2.ximgproc.BINARIZATION_WOLF)
    img_bin = binarize_img(cropImg)
    # cv2.imwrite("./crop/bin.jpg", img_bin)

    contours, _ = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    filtered_contour_center = []
    filtered_contour_boxes = []
    for contour in contours:
        contour_box = cv2.boundingRect(contour)
        contour_area = cv2.contourArea(contour)
        is_character_contour = contour_area > 100
        if is_character_contour:
            filtered_contours.append(contour)
            filtered_contour_center.append([contour_box[0] + contour_box[2] / 2, contour_box[1] + contour_box[3] / 2])
            filtered_contour_boxes.append(contour_box)
    text_boxes = []

    if len(filtered_contours) >= 2:
        filtered_contour_boxes = np.array(filtered_contour_boxes) # shape nx4
        filtered_contour_boxes[:, 2] += filtered_contour_boxes[:, 0]
        filtered_contour_boxes[:, 3] += filtered_contour_boxes[:, 1]

        right_top_to_left_top_dist = (filtered_contour_boxes[:, [0]] - filtered_contour_boxes[:, [2]].T) ** 2 +\
                                    (filtered_contour_boxes[:, [1]] - filtered_contour_boxes[:, [1]].T) ** 2
        right_bottom_to_right_top_dist = (filtered_contour_boxes[:, [2]] - filtered_contour_boxes[:, [2]].T) ** 2 +\
                                        (filtered_contour_boxes[:, [3]] - filtered_contour_boxes[:, [1]].T) ** 2

        # is_same_group = center_distance < 80 # 100 ** 2 + 50 ** 2
        is_same_group = (right_top_to_left_top_dist < threshold ** 2) | (right_bottom_to_right_top_dist < threshold ** 2)
        is_same_group = csr_array(is_same_group)
        n_components, labels = connected_components(is_same_group, directed=False)

        for label_idx in range(n_components):
            group_contours = [filtered_contours[i] for i in range(len(filtered_contours)) if labels[i] == label_idx]
            # color = np.random.choice(range(256), 3).tolist()
            group_box = cv2.boundingRect(np.concatenate(group_contours))
            text_boxes.append(group_box)
            # cv2.rectangle(cropImg, (group_box[0], group_box[1]), (group_box[0] + group_box[2], group_box[1] + group_box[3]), (0, 255, 0), 3)

    # detect by paddleocr
    result = ocr.ocr(cropImg, cls=False, rec=False)
    if result[0] is not None:
        for text_box in result[0]:
            text_box = np.array(text_box)
            text_box = text_box.astype(np.int32).reshape((-1, 1, 2))
            text_boxes.append(cv2.boundingRect(text_box))

    # tensor_boxes = torch.Tensor(text_boxes)
    # tensor_boxes[:, 2] += tensor_boxes[:, 0]
    # tensor_boxes[:, 3] += tensor_boxes[:, 1]
    # scores = [1.0] * len(tensor_boxes)
    # scores = torch.Tensor(scores)
    # roi_indices = nms(tensor_boxes, scores, 0.5).numpy()
    roi_boxes = []
    roi_indices = list(range(len(text_boxes)))
    for idx in roi_indices:
        x1, y1, x2, y2 = text_boxes[idx]
        if x2 < 40 or y2 < 40:
            continue
        x2 += x1
        y2 += y1
        # roi_boxes.append([x1, y1, x2, y2])
        roi_boxes.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    return [roi_boxes]