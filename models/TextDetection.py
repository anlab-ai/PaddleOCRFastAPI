import numpy as np
import cv2
import math

from scipy.spatial.distance import cdist
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components
import torch
from torchvision.ops import nms, box_iou
from shapely.geometry import Polygon
from shapely import intersection, intersects


def binarize_img(cropImg: cv2.Mat):
    h, s, v = cv2.split(cv2.cvtColor(cropImg, cv2.COLOR_BGR2HSV))
    cv2.imwrite("../results/hue/hue.jpg", s)
    _, img_bin = cv2.threshold(s, 20, 255, cv2.THRESH_BINARY)
    return img_bin

def bb_intersection_over_union(boxA, boxB):
    # # determine the (x, y)-coordinates of the intersection rectangle
    # xA = max(boxA[0], boxB[0])
    # yA = max(boxA[1], boxB[1])
    # xB = min(boxA[2], boxB[2])
    # yB = min(boxA[3], boxB[3])
    # # compute the area of intersection rectangle
    # interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # return interArea
    x1, y1, x2, y2 = boxA
    polygon1 = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    polygon1 = Polygon(polygon1)
    
    x1, y1, x2, y2 = boxB
    polygon2 = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    polygon2 = Polygon(polygon2)
    if intersects(polygon1, polygon2):
        return intersection(polygon1, polygon2).area / polygon1.area
    return 0

def compute_intersection_numpy(boxes1, boxes2):
    intersection = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for idx in range(len(boxes1)):
        for jdx in range(len(boxes2)):
            box1 = boxes1[idx]
            box2 = boxes2[jdx]
            intersection[idx, jdx] = bb_intersection_over_union(box1, box2)
    return intersection

def remove_box_by_intersect_ratio(boxes):
    tensor_boxes = np.array(boxes)
    intersection = compute_intersection_numpy(tensor_boxes, tensor_boxes)

    for idx in range(len(boxes)):
        intersection[idx, idx] = 0

    print(intersection)

    rows, cols = np.where(intersection > 0.6)
    remove_indices = list(set(rows.tolist()))
    indices = [idx for idx in range(len(boxes)) if idx not in remove_indices]
    return indices


def detect_chalk_text(cropImg: cv2.Mat, ocr, threshold=100, height_threshold=30):
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
        filtered_contour_cy = (filtered_contour_boxes[:, [3]] + filtered_contour_boxes[:, [1]]) / 2

        box_left = filtered_contour_boxes[:, [0]]
        box_top = filtered_contour_boxes[:, [1]]
        box_right = filtered_contour_boxes[:, [2]]
        box_bottom = filtered_contour_boxes[:, [3]]
        
        right_top_to_left_top_dist = np.minimum((box_left - box_right.T) ** 2 + (box_top - box_top.T) ** 2,
                                                (box_right - box_left.T) ** 2 + (box_top - box_top.T) ** 2)
        right_bottom_to_right_top_dist = np.minimum((box_right - box_right.T) ** 2 + (box_bottom - box_top.T) ** 2,
                                                    (box_right - box_right.T) ** 2 + (box_top - box_bottom.T) ** 2)
        height_distance = np.abs(filtered_contour_cy - filtered_contour_cy.T)
        is_same_group = (right_top_to_left_top_dist < threshold ** 2) | (right_bottom_to_right_top_dist < 180 ** 2)
        is_same_group = is_same_group & (height_distance < height_threshold)

        is_same_group = csr_array(is_same_group)
        n_components, labels = connected_components(is_same_group, directed=False)

        for label_idx in range(n_components):
            group_contours = [filtered_contours[i] for i in range(len(filtered_contours)) if labels[i] == label_idx]
            group_box = cv2.boundingRect(np.concatenate(group_contours))
            text_boxes.append(group_box)

    # detect by paddleocr
    result = ocr.ocr(cropImg, cls=False, rec=False)
    # result = ocr.ocr(cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR), cls=False, rec=False)

    # group box
    box_left, box_right, box_top, box_bottom = [], [], [], []
    box_width, box_height = [], []
    paddle_text_boxes = []
    if result[0] is not None:
        for text_box in result[0]:
            text_box = np.array(text_box)
            text_box = text_box.astype(np.int32).reshape((-1, 1, 2))
            left, top, width, height = cv2.boundingRect(text_box)
            paddle_text_boxes.append((left, top, width, height))

    for left, top, width, height in paddle_text_boxes:
        box_left.append(left)
        box_right.append(left+width)
        box_top.append(top)
        box_bottom.append(top+height)
        box_width.append(width)
        box_height.append(height)

    box_left = np.array(box_left).reshape((-1, 1))
    box_right = np.array(box_right).reshape((-1, 1))
    box_top = np.array(box_top).reshape((-1, 1))
    box_bottom = np.array(box_bottom).reshape((-1, 1))
    box_width = np.array(box_width).reshape((-1, 1))
    box_height = np.array(box_height).reshape((-1, 1))
    
    is_same_group_vertical = (np.abs(box_height - box_height.T) < 300) &\
                             ((np.abs(box_right - box_left.T) < threshold) | (np.abs(box_left - box_right.T) < threshold)) &\
                             (np.abs(box_bottom - box_top.T) < 500)
    is_same_group_horizontal = (np.abs(box_width - box_width.T) < 300) &\
                               ((np.abs(box_bottom - box_top.T) < threshold) | (np.abs(box_top - box_bottom.T) < threshold)) &\
                               ((np.abs(box_right - box_left.T) < 300) | (np.abs(box_left - box_right.T) < 300))
    is_same_group = is_same_group_vertical | is_same_group_horizontal
    is_same_group = csr_array(is_same_group)
    n_components, labels = connected_components(is_same_group, directed=False)
    group_text_boxes = []
    for label_idx in range(n_components):
        group_box = [paddle_text_boxes[i] for i in range(len(paddle_text_boxes)) if labels[i] == label_idx]
        if len(group_box) > 1:
            group_box = np.array(group_box)
            group_box[:, 2] += group_box[:, 0]
            group_box[:, 3] += group_box[:, 1]
            box_left, box_top, box_right, box_bottom = group_box[:, 0].min(), group_box[:, 1].min(), group_box[:, 2].max(), group_box[:, 3].max()
            box_width = box_right - box_left
            box_height = box_bottom - box_top
            # text_boxes.append((box_left, box_top, box_width, box_height))
            group_text_boxes.append((box_left, box_top, box_width, box_height))
        else:
            # text_boxes.extend(group_box)
            group_text_boxes.extend(group_box)

    # for x1, y1, x2, y2 in text_boxes:
    #     x1, y1, x2, y2 = list(map(int, (x1, y1, x2, y2)))
    #     cv2.rectangle(cropImg, (x1, y1), (x2+x1, y2+y1), (255, 0, 0), 3)

    # for x1, y1, x2, y2 in group_text_boxes:
    #     x1, y1, x2, y2 = list(map(int, (x1, y1, x2, y2)))
    #     cv2.rectangle(cropImg, (x1, y1), (x2+x1, y2+y1), (0, 0, 255), 3)
    # import time
    # cv2.imwrite(f"/home/hieu/hieunm/Paddle_parseq/results/detect_step/{time.time()}.jpg", cropImg)
    # print("Text boxes: ", text_boxes)
    # print("Group text boxes", group_text_boxes)

    if len(group_text_boxes) > 0 and len(text_boxes) > 0:
        filtered_text_boxes = []
        filtered_text_boxes.extend(group_text_boxes)
        group_tensor = np.array(group_text_boxes)
        if len(group_text_boxes) == 1:
            group_tensor = group_tensor.reshape((1, -1))
        group_tensor[:, 2] += group_tensor[:, 0]
        group_tensor[:, 3] += group_tensor[:, 1]
        
        text_tensor = np.array(text_boxes)
        if len(text_tensor) == 1:
            text_tensor = text_tensor.reshape((1, -1))
        text_tensor[:, 2] += text_tensor[:, 0]
        text_tensor[:, 3] += text_tensor[:, 1]
        
        intersection = compute_intersection_numpy(text_tensor, group_tensor)
        rows, cols = np.where(intersection > 0.7)
        rows = list(set(rows.tolist()))
        # for row_idx in rows:
        for idx in range(len(text_boxes)):
            if idx in rows:
                continue
            filtered_text_boxes.append(text_boxes[idx])
    else:
        filtered_text_boxes = text_boxes
    
    text_boxes = text_boxes + group_text_boxes

    roi_boxes = []
    roi_indices = list(range(len(filtered_text_boxes)))
    for idx in roi_indices:
        x1, y1, x2, y2 = filtered_text_boxes[idx]
        if (x2 < 40) or (y2 < 40) or (x2 / cropImg.shape[1] > 0.95) or (y2 / cropImg.shape[0] > 0.95):
            continue
        x2 += x1
        y2 += y1
        # roi_boxes.append([x1, y1, x2, y2])
        roi_boxes.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    
    return [roi_boxes]