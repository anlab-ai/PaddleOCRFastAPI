# -*- coding: utf-8 -*-

import time
from typing import List, Set
import json
import math
import argparse
import torch
from shapely.geometry import Polygon, Point

from pydantic import BaseModel
from paddleocr import PaddleOCR, draw_ocr
from utils.ImageHelper import *
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args
from models.TextDetection import detect_chalk_text
from torchvision.ops import nms
from PIL import Image
import io
import cv2
import re


class OCRModel(BaseModel):
    coordinate: List  # 图像坐标
    result: Set


class Base64PostModel(BaseModel):
    base64_str: str  # base64字符串
    
def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


class ImageReader():

    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=False, lang='japan', 
                             rec_model_dir="./chalk_font_hwjp_number_PP-OCRv3_inference", 
                             rec_char_dict_path="./chalk_font_hwjp_number_PP-OCRv3_inference/dict.txt",
                             det_model_dir="./paddle_models/det/red_chalk_PP-OCR_v3_det_inference/Student")
        parser = argparse.ArgumentParser()
        
        # parser.add_argument('--images', nargs='+', help='Images to read')
        parser.add_argument('--device', default='cpu')
        self.args, unknown = parser.parse_known_args()
        kwargs = {} #parse_model_args(unknown)
        # kwargs["model"] = dict()
        # kwargs['model']['charset_test'] = "0123456789"
        # print(kwargs)
        # print(f'Additional keyword arguments: {kwargs}')
        # self.model_plate_no = load_from_checkpoint('parseq_rec_model/parseq_plate_no_2024_09_13.ckpt', **kwargs).eval().to(self.args.device)
        self.model = load_from_checkpoint('parseq_rec_model/parseq-2024_05_19.ckpt', **kwargs).eval().to(self.args.device)
        # self.model_writer_1 = load_from_checkpoint('parseq_rec_model/parseq_writer_1.ckpt', **kwargs).eval().to(self.args.device)
        # print(f'model_writer_1: parseq_rec_model/parseq_writer_1.ckpt')
        self.img_transform = SceneTextDataModule.get_transform(self.model.hparams.img_size)
        self.max_length_text = 10
        

    def DetectTextBox(self, imageFileBytes):
        img = bytes_to_ndarray(imageFileBytes)
        formRatio = 480.0 / img.shape[1]
        img = cv2.resize(img, (0,0), fx=formRatio, fy=formRatio)
        result = self.ocr.ocr(img=img, cls=False, rec=False)
        for i in range(len(result)):
            boxes = result[i]
            for j in range(len(boxes)):
                box = boxes[j]
                for k in range(len(box)):
                    point = box[k]
                    point[0] /= formRatio
                    point[1] /= formRatio
                    box[k] = point
                boxes[j] = box
            result[i] = boxes
 
        print("result: ", result)
        return result
    
    def ReadImageWithPos(self, imageFileBytes, configs, items):
        img = bytes_to_ndarray(imageFileBytes)
        orgImg = img.copy()
        drawImg = orgImg.copy()
        boxes = items[0]
        for i in range(len(boxes)):
            boxes[i] = (quad_coords_to_xyxy(boxes[i]))
        boxes = mergeLine(boxes)

        txts = []
        origBoxes = []
        for i in range(len(boxes)):
            x_min,y_min,x_max,y_max = boxes[i]
            w,h = x_max-x_min,y_max-y_min
            externRatio = 0.1
            x = max(0, x_min - int(w*externRatio*0.5))
            y = max(0, y_min - int(h*externRatio*0.5))
            w += int(w*externRatio)
            origBoxes.append([int(x),int(y),int((x + w)),int((y + h))])
            textImg = orgImg[origBoxes[i][1]:origBoxes[i][3], origBoxes[i][0]:origBoxes[i][2]]
            
            grayImg = cv2.cvtColor(textImg, cv2.COLOR_BGR2GRAY)
            T, binImg = cv2.threshold(grayImg, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            textImg = binImg
            result = self.ocr.ocr(img=textImg, cls=False, det=False)
            print("result: ", result)
            txts.append(result[0][0][0])
        result_txts = txts
        result_boxs = origBoxes
        if len(configs) > 0:
            result_txts = []
            result_boxs = []
            numberDigits = configs["total_digit"]
            numberDigitBeforeDot = configs["digit_before_dot"]
            for i in range(len(txts)):
                text = txts[i]
                text = re.sub("[\D]", "", text)
                if len(text) == numberDigits or numberDigits == 0:
                    if numberDigitBeforeDot > 0 and numberDigitBeforeDot < len(text):
                        text = text[:numberDigitBeforeDot] + '.' + text[numberDigitBeforeDot:]
                    result_txts.append(text)
                    result_boxs.append(origBoxes[i])
        drawImg = drawResult(drawImg, result_boxs, result_txts)
        array = cv2.cvtColor(np.array(drawImg), cv2.COLOR_RGB2BGR)
        im_show = Image.fromarray(array, mode="RGB")
        bytes_image = io.BytesIO()
        im_show.save(bytes_image, format='PNG')

        return bytes_image.getvalue()
    
    def ReadImageWithMode(self, imageFileBytes, mode, infos: str):
        if mode == '1':
            full_screen = [[0.3575, 0.05], [1-0.3575, 0.05], [1-0.025, 1-0.05], [0.025, 1-0.05]]
            # 9 box
            screen_boxes = [
                [[0.26, 0.3], [0.32, 0.35], [0.26, 0.52], [0.20, 0.46]],
                [[0.135, 0.68], [0.168, 0.72], [0.09, 0.936], [0.061, 0.8769]],
                [[0.398, 0.41], [0.588, 0.41], [0.588, 0.53], [0.398, 0.53]],
                [[0.398, 0.58], [0.588, 0.58], [0.588, 0.70], [0.398, 0.70]],
                [[0.14, 0.84], [0.36, 0.84], [0.36, 0.937], [0.14, 0.937]],
                [[0.66, 0.84], [0.86, 0.84], [0.86, 0.937], [0.66, 0.937]],
                [[0.68, 0.34], [0.735, 0.29], [0.795, 0.4499], [0.74, 0.5019]],
                [[0.76, 0.57], [0.82, 0.52], [0.88, 0.68], [0.819, 0.73]],
                [[0.83, 0.746], [0.888, 0.70], [0.96, 0.8969], [0.903, 0.932]]
            ]
            left_idx = 0
            right_idx = 6
        elif mode == '2':
            full_screen = [[0.25, 0], [1-0.25, 0], [1-0.025, 1-0.05], [0.025, 1-0.05]]
            # 5 box
            screen_boxes = [
                [[0.3, 0.1], [0.1, 0.936], [0.042, 0.8769], [0.2361, 0.068]],
                [[0.39, 0.1], [0.62, 0.1], [0.62, 0.74], [0.39, 0.74]],
                [[0.135, 0.8], [0.36, 0.8], [0.36, 0.95], [0.135, 0.95]],
                [[0.63, 0.8], [0.86, 0.8], [0.86, 0.95], [0.63, 0.95]],
                [[0.893, 0.932], [0.701, 0.1], [0.765, 0.068], [0.96, 0.8969]]
            ]
            left_idx = 0
            right_idx = 4
        infos = json.loads(infos)
        total_digit = int(infos.get("total_digit", 0))
        digit_before_dot = int(infos.get("digit_before_dot", 0))
        use_rotate_on_every_image = True
        use_extend_on_every_image = True
            
        print(infos)
        
        # with open(position_path, "r") as f:
        #     positions = json.load(f)["data"]
        img = bytes_to_ndarray(imageFileBytes)
        drawImg = img.copy()
        img_H, img_W = img.shape[:2]
        positions = np.array(screen_boxes)
        print(positions.shape)
        positions[:, :, 0] *= img_W
        positions[:, :, 1] *= img_H
        positions = positions.astype(np.int32)
        
        left_position = positions[left_idx]
        right_position = positions[right_idx]
        left_polygon = Polygon(left_position)
        right_polygon = Polygon(right_position)


        #crop and rotate text image
        list_box = []
        images = []
        txts = []
        for i in range(len(positions)):
            # p1 = [positions[i][0]["x"],positions[i][0]["y"]]
            # p2 = [positions[i][1]["x"],positions[i][1]["y"]]
            # p3 = [positions[i][2]["x"],positions[i][2]["y"]]
            # p4 = [positions[i][3]["x"],positions[i][3]["y"]]
            p1 = positions[i][0]
            p2 = positions[i][1]
            p3 = positions[i][2]
            p4 = positions[i][3]
            x_min,y_min,x_max,y_max = quad_coords_to_xyxy([p1,p2,p3,p4])
            
            
            define_box = [p1, p2, p3, p4]
            define_box = np.array(define_box, np.int32).reshape((-1, 1, 2))
            x, y, width, height = cv2.boundingRect(define_box)
            rotated_rect = cv2.minAreaRect(define_box)
            is_rotated = rotated_rect[2] >= 10 and rotated_rect[2] <= 80

            if is_rotated:
                line1 = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
                line2 = math.sqrt((p4[0] - p1[0]) ** 2 + (p4[1] - p1[1]) ** 2)
                if line1 >= line2:
                    new_points = [[0, 0], [line1, 0], [line1, line2], [0, line2]]
                else:
                    new_points = [[0, line1], [0, 0], [line2, 0], [line2, line1]]
                old_pts = np.array([p1, p2, p3, p4], dtype=np.float32).reshape((-1, 1, 2))
                new_pts = np.array(new_points, dtype=np.float32).reshape((-1, 1, 2))
                # M = cv2.getPerspectiveTransform(old_pts, new_pts)
                M, _ = cv2.findHomography(old_pts, new_pts)
                cropImg = cv2.warpPerspective(img, M, (int(max(line1, line2)), int(min(line1, line2))))
            else:
                cropImg = img[y:y+height, x:x+width]

            # cv2.imwrite(f"./crop/crop/{i}.jpg", cropImg)
            result = detect_chalk_text(cropImg, self.ocr)

            if len(result[0]) == 0:
                images.append(self.img_transform(Image.fromarray(cropImg, 'RGB')))
                # import time
                # cv2.imwrite(f"./crop/{time.time()}.jpg", cropImg)
                # images.append(cropImg)
                list_box.append((x_min,y_min,x_max,y_max))
            else:
                for box in result[0]:
                    x,y,x_m,y_m = quad_coords_to_xyxy(box)
                    # x = int(x/scale)
                    # y = int(y/scale)
                    # x_m = int(x_m/scale)
                    # y_m = int(y_m/scale)
                    textImg = cropImg[int(y):int(y_m), int(x):int(x_m)]
                    text_width, text_height = x_m - x, y_m - y
                    # textImg = cropImg[int(y-text_height*0.1):int(y_m+text_height*0.1), int(x-text_width*0.1):int(x_m+text_width*0.1)]
                    # if min(textImg.shape) == 0:
                    #     continue
                    images.append(self.img_transform(Image.fromarray(textImg, 'RGB')))
                    # cv2.imwrite(f"./crop/text/{time.time()}.jpg", textImg)
                    # cv2.imwrite(f"./crop/{time.time()}.jpg", textImg)
                    # images.append(textImg)
                    if is_rotated:
                        # cv2.imwrite("cropImg.jpg", cropImg)
                        # print(box)
                        inv_box = [[x, y], [x_m, y], [x_m, y_m], [x, y_m]]
                        inv_box = np.array(inv_box, dtype=np.float32).reshape((-1, 1, 2))
                        inv_box = cv2.perspectiveTransform(inv_box, np.linalg.inv(M))
                        # inv_box[:, :, 0] += x_min
                        # inv_box[:, :, 1] += y_min
                        inv_box = inv_box.astype(np.int32)
                        x1, y1, x2, y2 = quad_coords_to_xyxy(inv_box.squeeze(1).tolist())
                        list_box.append((x1, y1, x2, y2))
                    else:
                        for i in range(len(box)):
                            box[i][0] += x_min
                            box[i][1] += y_min
                        list_box.append((x_min + x,y_min + y,x_min + x_m,y_min+y_m))

        result = detect_chalk_text(img, self.ocr)
        for box in result[0]:
            x, y, x_m, y_m = quad_coords_to_xyxy(box)
            text_width, text_height = x_m - x, y_m - y

            if left_polygon.contains(Point((x+x_m)/2, (y+y_m)/2)):
                x1, y1 = int(max(0, x-text_width*0.5)), int(max(0, y-text_height*0.5))
                x2, y2 = int(x_m + text_width*0.5), int(y_m + text_height*0.3)
                textImg = img[y1:y2, x1:x2]
                textImg = rotate_image(textImg, 135)
                r_h, r_w = textImg.shape[:2]
                # cv2.imwrite(f"./crop/text/{time.time()}.jpg", textImg)
                textImg = textImg[r_h//10:-r_h//2]
            elif right_polygon.contains(Point((x+x_m)/2, (y+y_m)/2)):
                x1, y1 = int(max(0, x-text_width*0.5)), int(max(0, y-text_height*0.5))
                x2, y2 = int(x_m + text_width*0.5), int(y_m + text_height*0.5)
                textImg = img[y1:y2, x1:x2]
                textImg = rotate_image(textImg, -135)
            else:
                textImg = img[int(y):int(y_m), int(x):int(x_m)]
            images.append(self.img_transform(Image.fromarray(textImg, 'RGB')))
            list_box.append((x, y, x_m, y_m))
            
            if (use_rotate_on_every_image):
                textImg = img[int(y):int(y_m), int(x):int(x_m)]
                images.append(self.img_transform(Image.fromarray(textImg, 'RGB')))
                list_box.append((x, y, x_m, y_m))
                
            if (use_extend_on_every_image):
                x1, y1 = int(max(0, x-text_width*0.5)), int(max(0, y-text_height*0.5))
                x2, y2 = int(x_m + text_width*0.5), int(y_m + text_height*0.3)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(x1, img_W), min(y2, img_H)
                textImg = img[y1:y2, x1:x2]
                images.append(self.img_transform(Image.fromarray(textImg, 'RGB')))
                list_box.append((x, y, x_m, y_m))

        if len(images) > 0:
            images = torch.stack(images).to(self.args.device)
            with torch.no_grad():
                p = self.model(images)
                p =  torch.softmax(p, dim=2)
                p[:, :, 11:74] = 0
                p[:, :, 75:76] = 0
                p[:, :, 77:] = 0
                pred, p = self.model.tokenizer.decode(p, text_threshold=0.5)
            scores = ([s.cpu().mean().item() for s in p])
            texts = pred
            # std_probs = []
            # for img_id in range(len(scores)):
            #     text = texts[img_id]
            #     prob = p[img_id]
            #     valid_token_ids = [token_id for token_id in range(len(text))
            #                        if prob[token_id] > 0.5]
            #     prob = [prob[token_id] for token_id in valid_token_ids]
            #     std_prob = np.array(prob).std()
            #     std_probs.append(std_prob)
            #     text = [text[token_id] for token_id in valid_token_ids]
            #     text = "".join(text)
            # std_probs = [np.array(s).std() for s in p]
            print(scores)
        print("output texts: ", texts)

        tensor_boxes = torch.Tensor(list_box)
        scores = torch.Tensor(scores)
        roi_indices = nms(tensor_boxes, scores, 0.3).numpy()
        # print(tensor_boxes.shape, scores.shape)
        # roi_indices = list(range(len(scores)))

        full_screen = [[x * img_W, y * img_H] for (x, y) in full_screen]
        screen_polygon = Polygon(full_screen)
        for idx in roi_indices:
            text = texts[idx]
            x_min, y_min, x_max, y_max = list_box[idx]
            
            # if scores[idx] < 0.7:
            #     continue
            # if std_probs[idx] > 0.1:
            #     continue
            if scores[idx] < 0.9:
                continue
            if len(text) > self.max_length_text:
                continue
            is_contain_number = any([c for c in text if c.isdigit()])
            if not is_contain_number:
                continue
            if x_max - x_min > img_W * 0.3 or y_max - y_min > img_H * 0.3:
                continue

            point = Point((x_min+x_max)/2 - (x_max-x_min)*0.1, (y_min+y_max)/2 - (y_max-y_min)*0.1)
            if not screen_polygon.contains(point) and scores[idx] < 0.95:
                continue
            
            if digit_before_dot > 0:
                if "." or "," in text:
                    output_text = "".join([char for char in text if char.isdigit()])
                    if len(output_text) > digit_before_dot:
                        output_text = output_text[:digit_before_dot] + "." + output_text[digit_before_dot:]
                        text = output_text
            cv2.rectangle(drawImg, (int(x_min),int(y_min)), (int(x_max),int(y_max)), (0, 255, 0), 10)
            cv2.putText(drawImg, text, (int(x_min),int(y_min)), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 255), img_H//160,
                        bottomLeftOrigin=False)
        
        drawImg = cv2.resize(drawImg, (0,0), fx=0.5, fy=0.5)
        drawImg = cv2.cvtColor(drawImg, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(drawImg)
        bytes_image = io.BytesIO()
        pil_image.save(bytes_image, format='PNG')

        return bytes_image.getvalue()
            