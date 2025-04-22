# -*- coding: utf-8 -*-

import base64

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def base64_to_ndarray(b64_data: str):
    """base64转numpy数组

    Args:
        b64_data (str): base64数据

    Returns:
        _type_: _description_
    """
    image_bytes = base64.b64decode(b64_data)
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    image_np2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    return image_np2


def bytes_to_ndarray(img_bytes: str):
    """字节转numpy数组

    Args:
        img_bytes (str): 图片字节

    Returns:
        _type_: _description_
    """
    image_array = np.frombuffer(img_bytes, dtype=np.uint8)
    image_np2 = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image_np2

def xyxyxyxy2xywh(bboxes):
    new_bboxes = np.zeros([len(bboxes), 4])
    new_bboxes[:, 0] = bboxes[:, 0::2].min()  # x1
    new_bboxes[:, 1] = bboxes[:, 1::2].min()  # y1
    new_bboxes[:, 2] = bboxes[:, 0::2].max() - new_bboxes[:, 0]  # w
    new_bboxes[:, 3] = bboxes[:, 1::2].max() - new_bboxes[:, 1]  # h
    return new_bboxes

def xyxy2xywh(bboxes):
    new_bboxes = np.empty_like(bboxes)
    new_bboxes[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2  # x center
    new_bboxes[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2  # y center
    new_bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]  # width
    new_bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]  # height
    return new_bboxes

def quad_coords_to_xyxy(quad_coords):
    x_values = [x for x, _ in quad_coords]
    y_values = [y for _, y in quad_coords]

    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)

    # Calculate the center (x_center, y_center) and width (w) and height (h)
    # x_center = (x_min + x_max) / 2
    # y_center = (y_min + y_max) / 2
    # w = x_max - x_min
    # h = y_max - y_min
    return x_min,y_min,x_max,y_max

def drawResult(img, boxes, txts):
    print("text: ", txts)
    d = ImageDraw.Draw(Image.fromarray(img))
    fnt = ImageFont.truetype('fonts/NotoSans-Regular.ttf', 5)
    for i in range(len(txts)):
        x_min,y_min,x_max,y_max = boxes[i]
        img = cv2.rectangle(img, (int(x_min),int(y_min)), (int(x_max),int(y_max)), (0, 255, 0), 2)
        img = cv2.putText(img, txts[i], (int(x_min),int(y_min)), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
        # info = txts[i]
        # info.encode("utf-8")
        # d.text([0,0], info, fill=(255, 0, 0), font=fnt)

    return img
def sortTextBox(boxes):
    boxes = sorted(boxes, key=lambda rect: (rect[0], rect[1]))
    sortedBoxs = []
    for i in range(len(boxes)):
        added = False
        for j in range(len(sortedBoxs)):
            y,h = sortedBoxs[j][0][1], sortedBoxs[j][0][3]-sortedBoxs[j][0][1]
            if (boxes[i][1] >= y - h / 2) and (boxes[i][1] < y + h / 2):
                sortedBoxs[j].append(boxes[i])
                added = True
                break
        if not added:
            sortedBoxs.append([boxes[i]])

    return sortedBoxs

def getBoundingBoxOfListBox(boxes):
    x_values = []
    y_values = []
    for j in range(len(boxes)):
        x_values.append(boxes[j][0])
        x_values.append(boxes[j][2])
        y_values.append(boxes[j][1])
        y_values.append(boxes[j][3])
    print("x_values: ", x_values)
    print("y_values: ", y_values)

    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    return (x_min, y_min, x_max, y_max)

def mergeLine(boxes):
    print("boxes: ", boxes)
    sortedBoxs = sortTextBox(boxes)
    lines = []
    for i in range(len(sortedBoxs)):
        if len(sortedBoxs[i]) < 1:
            continue
        print("sortedBoxs {}: {}".format(i, sortedBoxs[i]))
        
        group = []
        for k in range(len(sortedBoxs[i])):
            if len(group) == 0:
                group.append(sortedBoxs[i][k])
                continue
            else:
                distance = sortedBoxs[i][k][0] - group[-1][2]
                print("distance {}: {}".format(k, distance))
                if distance <= max(sortedBoxs[i][k][2]-sortedBoxs[i][k][0], group[-1][2]-group[-1][0])/2:
                    group.append(sortedBoxs[i][k])
                    continue
                lines.append(getBoundingBoxOfListBox(group))
                group = []
                group.append(sortedBoxs[i][k])


        if len(group) > 0:
            lines.append(getBoundingBoxOfListBox(group))
            group = []
            
    return lines




import cv2
import numpy as np

def draw_vertical_text(drawImg, text, position, font, font_size, color, thickness, ):
    """
    Draw vertical text on an image by rotating it 90 degrees counterclockwise.
    
    Args:
        drawImg: Input image (BGR format, NumPy array).
        text: String to draw.
        position: Tuple (x_min, y_min) for text placement.
        font: OpenCV font (e.g., cv2.FONT_HERSHEY_SIMPLEX).
        font_size: Font scale factor.
        color: Text color in BGR format (e.g., (0, 0, 255) for red).
        thickness: Text thickness.
        **kwargs: Additional arguments for cv2.putText (e.g., lineType).

    Returns:
        Image with vertical text drawn.
    """
    
    # Get text size for temporary image
    text_size, _ = cv2.getTextSize(text, font, font_size, thickness)
    text_w,text_h = text_size
    temp_img = np.zeros((text_size[1], text_size[0], 3), dtype=np.uint8)  # BGR temp image

    # Draw text on temporary image
    cv2.putText(temp_img, text, (0, text_size[1]), font, font_size, color, thickness)

    # Rotate the temporary image 90 degrees counterclockwise
    rotated = cv2.rotate(temp_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Calculate position and size
    x_min, y_min,y_max = position
    h, w = rotated.shape[:2]

    # Check bounds to avoid out-of-bounds errors
    if x_min + w <= drawImg.shape[1] and y_min + h <= drawImg.shape[0]:
        # Blend rotated text onto the original image
        # Create a mask for non-zero pixels (text pixels) in any channel
        mask = np.any(rotated != 0, axis=2)  # True where any channel is non-zero
        # ensure h_top + h_bottom = h due to image copy must match shape
        h_top = h//2
        h_bottom = h-h//2
        # Copy text pixels to the original image using the mask
        drawImg[(y_min+y_max)//2-h_top:(y_min+y_max)//2+h_bottom, x_min-text_h:x_min+w-text_h][mask] = rotated[mask]
    else:
        print(f"Warning: Text at ({x_min}, {y_min}) exceeds image bounds ({drawImg.shape[1]}, {drawImg.shape[0]}).")

    return drawImg


def draw_horizonal_text(drawImg,
                        text,
                        position,
                        font,
                        font_size,
                        color,
                        thickness,
                        bottomLeftOrigin=False):
    x_min,y_min,x_max = position
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_size, thickness)
    if x_min + text_width> drawImg.shape[1]:
        x_min = drawImg.shape[1] - text_width
    if y_min + text_height > drawImg.shape[0]:
        y_min = drawImg.shape[0] - text_height
    cv2.putText(drawImg, text, (int((x_max+x_min-text_width)/2),int(y_min)), font, font_size, color, thickness,
						bottomLeftOrigin=bottomLeftOrigin)
    return drawImg