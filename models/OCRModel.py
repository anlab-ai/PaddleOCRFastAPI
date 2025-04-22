# -*- coding: utf-8 -*-

import time
from typing import List, Set
import json
import math
import argparse
import torch
from shapely.geometry import Polygon, Point
from torchvision.ops import box_iou

from pydantic import BaseModel
from paddleocr import PaddleOCR, draw_ocr
from utils.ImageHelper import *
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args
from models.TextDetection import detect_chalk_text, remove_box_by_intersect_ratio
from torchvision.ops import nms
from PIL import Image
import io
import cv2
import re
from loguru import logger

from ultralytics.utils.ops import xyxyxyxy2xywhr,xywhr2xyxyxyxy
from .sahi.sahi_phuoc import YOLO_SAHI
from .sahi.flip_img import Rotator
from .sahi.image_cropper import perspective_transform,clockwise_sort
from utils.ImageHelper import draw_vertical_text,draw_horizonal_text

class OCRModel(BaseModel):
	coordinate: List  # 图像坐标
	result: Set


class Base64PostModel(BaseModel):
	base64_str: str  # base64字符串
	
# def rotate_image(image, angle):
#   image_center = tuple(np.array(image.shape[1::-1]) / 2)
#   rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
#   result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
#   return result



# def flip_vertical(img):
	# h,w,_ = img.shape
	# src = np.array([
	# 			[0, 0],
	# 			[w - 1, 0],
	# 			[w - 1, h - 1],
	# 			[0, h - 1]
	# 		], dtype=np.float32)
 
	# dst = np.array([
	# 			[0, h - 1],
	# 			[w - 1, h - 1],
	# 			[w - 1, 0],
	# 			[0, 0],
	# 		], dtype=np.float32)
	# M = cv2.getPerspectiveTransform(src, dst)

	# Check if transformation matrix is valid
	# if M is None:
	# 	raise ValueError("Failed to compute perspective transform matrix")
		
	# # Apply perspective transform with border handling
	# warped = cv2.warpPerspective(
	# 	img, 
	# 	M, 
	# 	(w, h),
	# 	flags=cv2.INTER_LINEAR,
	# 	borderMode=cv2.BORDER_CONSTANT,
	# 	borderValue=(0, 0, 0)  # Black border for out-of-bounds areas
	# )
	# return warped
# def flip_left(img):
# 	h,w,_ = img.shape
# 	src = np.array([
# 				  	[0, 0],        # top-left
# 					[w - 1, 0],    # top-right
# 					[w - 1, h - 1], # bottom-right
# 					[0, h - 1] 
# 			], dtype=np.float32)
 
# 	dst = np.array([
# 				[0, w-1],
# 				[0,0],
# 				[h-1,0],
# 				[h-1,w-1],
				
# 			], dtype=np.float32)
# 	M = cv2.getPerspectiveTransform(src, dst)

# 	# Check if transformation matrix is valid
# 	if M is None:
# 		raise ValueError("Failed to compute perspective transform matrix")
		
# 	# Apply perspective transform with border handling
# 	warped = cv2.warpPerspective(
# 		img, 
# 		M, 
# 		(h, w),
# 		flags=cv2.INTER_LINEAR,
# 		borderMode=cv2.BORDER_CONSTANT,
# 		borderValue=(0, 0, 0)  # Black border for out-of-bounds areas
# 	)
# 	return warped
# def flip_right(img):

   
# 	# Get original dimensions
# 	h, w, _ = img.shape
	
# 	# Source points (original corners)
# 	src = np.array([
# 		[0, 0],        # top-left
# 		[w - 1, 0],    # top-right
# 		[w - 1, h - 1], # bottom-right
# 		[0, h - 1]      # bottom-left
# 	], dtype=np.float32)
	
# 	# Destination points for 90° clockwise rotation
# 	# Mapped to new dimensions (h, w)
# 	dst = np.array([
# 		[h - 1, 0],    # new top-right
# 		[h - 1, w - 1], # new bottom-right
# 		[0, w - 1],     # new bottom-left
# 		[0, 0]          # new top-left
# 	], dtype=np.float32)
	
# 	# Compute transformation matrix
# 	M = cv2.getPerspectiveTransform(src, dst)
 
# 	# Apply transform with new dimensions
# 	rotated = cv2.warpPerspective(
# 		img,
# 		M,
# 		(h, w),  # New width = old height, new height = old width
# 		flags=cv2.INTER_LINEAR,
# 		borderMode=cv2.BORDER_CONSTANT,
# 		borderValue=(0, 0, 0)
# 	)
# 	return rotated

# class Rotator:
# 	alpha = 1.7
# 	@staticmethod
# 	def get_rotate_images(original_image):
# 		h, w, _ = original_image.shape
# 		rotated_images = [
# 			original_image.copy(),
# 			flip_vertical(np.copy(original_image)),
# 			flip_left(np.copy(original_image)),
# 			flip_right(np.copy(original_image)),
# 		]
# 		return rotated_images

# def get_max_width_height(rect):
# 	"""Tính chiều rộng và chiều cao tối đa của hình chữ nhật"""
# 	widthA = np.linalg.norm(rect[2] - rect[3])  # BR - BL
# 	widthB = np.linalg.norm(rect[1] - rect[0])  # TR - TL
# 	maxWidth = int(max(widthA, widthB))

# 	heightA = np.linalg.norm(rect[1] - rect[2])  # TR - BR
# 	heightB = np.linalg.norm(rect[0] - rect[3])  # TL - BL
# 	maxHeight = int(max(heightA, heightB))

# 	return maxWidth, maxHeight
# def order_points(pts):
	
# 	"""Sắp xếp 4 điểm theo thứ tự: [TL, TR, BR, BL]"""
# 	pts = pts[np.argsort(pts[:, 0])]  # Sắp xếp theo tọa độ x
# 	left_pts, right_pts = pts[:2], pts[2:]
	
# 	# Xác định top-left và bottom-left theo y
# 	left_pts = left_pts[np.argsort(left_pts[:, 1])]
# 	right_pts = right_pts[np.argsort(right_pts[:, 1])]
	
# 	rect = np.array([left_pts[0], right_pts[0], right_pts[1], left_pts[1]], dtype=np.float32)
# 	return rect

# def crop_region(image, rect):

# 	"""Cắt ảnh theo bounding box của polygon và cập nhật tọa độ điểm"""
# 	x_min, y_min = np.min(rect, axis=0).astype(int)
# 	x_max, y_max = np.max(rect, axis=0).astype(int)
# 	cropped = image[y_min:y_max, x_min:x_max].copy()
	
# 	# Cập nhật lại tọa độ điểm cho ảnh đã cắt
# 	new_rect = rect - np.array([x_min, y_min]).reshape(1,2)
# 	return cropped, np.array(np.clip(new_rect, [0, 0], [image.shape[1], image.shape[0]]),np.float32)

# def perspective_transform(image, pts):
# 	"""
# 	Cắt và biến đổi phối cảnh một vùng trong ảnh thành hình chữ nhật.
# 	Args:
# 		image: Input image (numpy array)
# 		pts: Array of points defining the region to transform
# 	Returns:
# 		warped: Transformed rectangular image
# 	"""
# 	try:
# 		# Check if input is valid
# 		if image is None or pts is None:
# 			raise ValueError("Image or points cannot be None")
		
# 		if len(pts) != 4:
# 			raise ValueError("Exactly 4 points are required for perspective transform")
			
# 		# Convert points to numpy array if not already
# 		pts = np.array(pts, dtype=np.float32)
		
# 		# Order points and get dimensions
# 		polygons = order_points(pts)
# 		maxWidth, maxHeight = get_max_width_height(polygons)
		
# 		# Handle case where dimensions are invalid
# 		if maxWidth <= 0 or maxHeight <= 0:
# 			raise ValueError("Invalid dimensions calculated for transformation")
			
# 		# Crop the region with safety checks
# 		cropped_image, polygons = crop_region(image, polygons)
# 		if cropped_image is None or cropped_image.size == 0:
# 			raise ValueError("Failed to crop region from image")
			
# 		# Define destination points
# 		dst = np.array([
# 			[0, 0],
# 			[maxWidth - 1, 0],
# 			[maxWidth - 1, maxHeight - 1],
# 			[0, maxHeight - 1]
# 		], dtype=np.float32)
		
# 		# Calculate perspective transform matrix
# 		M = cv2.getPerspectiveTransform(polygons, dst)
		
# 		# Check if transformation matrix is valid
# 		if M is None:
# 			raise ValueError("Failed to compute perspective transform matrix")
			
# 		# Apply perspective transform with border handling
# 		warped = cv2.warpPerspective(
# 			cropped_image, 
# 			M, 
# 			(maxWidth, maxHeight),
# 			flags=cv2.INTER_LINEAR,
# 			borderMode=cv2.BORDER_CONSTANT,
# 			borderValue=(0, 0, 0)  # Black border for out-of-bounds areas
# 		)
		
# 		# Verify output
# 		if warped is None or warped.size == 0:
# 			raise ValueError("Perspective transformation failed")
			
# 		return warped
		
# 	except Exception as e:
# 		print(f"Error in perspective_transform: {str(e)}")
# 		return None

class ImageReader():

	def __init__(self):
		parser = argparse.ArgumentParser()
		
		# parser.add_argument('--images', nargs='+', help='Images to read')
		parser.add_argument('--device', default='cpu')
		self.args, unknown = parser.parse_known_args()
		kwargs = {} #parse_model_args(unknown)
		self.model = load_from_checkpoint('parseq_rec_model/parsseq_multi_direct.pt', **kwargs).eval().to(self.args.device)
		self.img_transform = SceneTextDataModule.get_transform(self.model.hparams.img_size)
		self.max_length_text = 10
		self.detection_model = YOLO_SAHI(
			model_path="parseq_rec_model/best.pt",
			confidence_threshold=0.35,
			device=self.args.device,
			slice_height = 840,
			slice_width = 840,
			overlap = 0.3,
			iou_merge_sahi = 0.4,
			
		)
		self.rotator = Rotator("/home/hieu/hieunm/Paddle_parseq/parseq_rec_model/best_cls_text_direct.pt")
	def recognize_single_text(self, image: np.ndarray):
		transform_image = [self.img_transform(Image.fromarray(image, "RGB"))]
		inputs = torch.stack(transform_image).to(self.args.device)
		with torch.no_grad():
			p = self.model(inputs)
			p =  torch.softmax(p, dim=2)
			# p[:, :, 11:73] = 0
			# p[:, :, 75:76] = 0
			# p[:, :, 77:] = 0
			text, p = self.model.tokenizer.decode(p)
		score = ([s.cpu().mean().item() for s in p])
		return text[0], score[0]

	def recognize_text(self, images: list[np.ndarray],x_y_mean = None):
		texts, scores = [], []
		#debug
		# img_save_folder = "/home/hieu/hieunm/Paddle_parseq/phuoc_tmp"
		# import time
		# import os
		for i,img_crop in enumerate(images):
			
			
			transform_image = self.rotator.get_rotate_images(img_crop,mean_points=(x_y_mean[i]))
			
			image = transform_image[0][0]
			#debug
			# debug_img = image.copy()
			pred, conf = self.recognize_single_text(image)
			texts.append(pred)
			scores.append(conf)

			#for debug
			# name = f"{pred}_{time.time()}.jpg"
			# path = os.path.join(img_save_folder,name)
			# cv2.imwrite(path,debug_img)

			# inputs = torch.stack(transform_image).to(self.args.device)
			# logger.info("Preprocess success")
			# with torch.no_grad():
			#     p = self.model(inputs)
			#     p =  torch.softmax(p, dim=2)
			#     # p[:, :, 11:73] = 0
			#     # p[:, :, 75:76] = 0
			#     # p[:, :, 77:] = 0
			#     text, p = self.model.tokenizer.decode(p)
			# score = ([s.cpu().mean().item() for s in p])
			# max_idx = np.argmax(score)
			# logger.info("Predict success")
			# texts.append(text[max_idx])
			# scores.append(score[max_idx])

		# inputs = torch.stack(transform_images).to(self.args.device)
		# logger.info("Load image success")
		# with torch.no_grad():
		#     p = self.model(inputs)
		#     p =  torch.softmax(p, dim=2)
		#     # p[:, :, 11:73] = 0
		#     # p[:, :, 75:76] = 0
		#     # p[:, :, 77:] = 0
		#     preds, p = self.model.tokenizer.decode(p)
		#     confs = ([s.cpu().mean().item() for s in p])
		#     logger.info("Predict success")

		# logger.info(f"Predict results: {preds}")

		# n_boxes = len(images)
		# for idx in range(n_boxes):
		#     text = preds[idx*4:idx*4+4]
		#     score = confs[idx*4:idx*4+4]
		#     max_idx = np.argmax(score)
		#     texts.append(text[max_idx])
		#     scores.append(score[max_idx])
		return texts, scores

	def ReadImageWithMode(self, imageFileBytes, mode, infos: str):
		logger.info("Start read image")
		# infos = json.loads(infos)
		# total_digit = int(infos.get("total_digit", 0))
		# digit_before_dot = int(infos.get("digit_before_dot", 0))
		total_digit = 0
		digit_before_dot = 0

		img = bytes_to_ndarray(imageFileBytes)
		
		drawImg = img.copy()
		img_H, img_W = img.shape[:2]
		logger.info("Start detection")
		dbscan_final_box, dbscan_final_confidences = self.detection_model.predict_from_path(img)
		logger.info("Detection done!")
		#mode 3: Filter all box has confidence < box_thresh (default 0.58)
		if str(mode) == '3':
			box_thresh = 0.58
			filtered = [(box, conf) for box, conf in zip(dbscan_final_box, dbscan_final_confidences) if conf > box_thresh]
			dbscan_final_box, dbscan_final_confidences = zip(*filtered) if filtered else ([], [])
	
		boxes = []
		images = []

		x_y_mean = []

		for box in dbscan_final_box:
			points = np.array(box, dtype=np.float32).reshape(-1, 1, 2)
			points = points.astype(np.int32)
			l, t, w, h = cv2.boundingRect(points)

			xywhr = xyxyxyxy2xywhr(np.array([box],dtype=np.float32))
			pts2 = xywhr2xyxyxyxy(xywhr).squeeze()
			pts2 = np.array(clockwise_sort(pts2))[::-1]
			x_y_mean.append((np.mean(pts2[:,0])/img_W,np.mean(pts2[:,1])/img_H))
			img_crop = perspective_transform(img,np.copy(pts2))
			# img_crop = img[t:t+h, l:l+w]
			
			images.append(img_crop)
			boxes.append([l, t, w, h])
			#phuoc debug
			# x_min, y_min, x_max, y_max = l, t, l+w, t+h
			# cv2.rectangle(drawImg, (int(x_min),int(y_min)), (int(x_max),int(y_max)), (0, 255, 0), 10)

		logger.info("Detection and crop done!")
		texts, scores = self.recognize_text(images,x_y_mean)
		logger.info("Recognition done")

		# print(texts,scores)
		if str(mode) == '3':
			text_threshold = 0.58
			filtered = [(text, conf,box) for text, conf,box in zip(texts, scores,boxes) if conf > text_threshold]
			texts, scores,boxes = zip(*filtered) if filtered else ([], [],[])

		print(f"Text: {texts}")
		print(f"Scores: {scores}")

		draw_boxes = []
		draw_texts = []

		font_size = img_H//1000
		thickness = img_H//700
		font = cv2.FONT_HERSHEY_SIMPLEX

		for idx in range(len(texts)):
			text = texts[idx]
			score = scores[idx]
			l, t, w, h = boxes[idx]
			x_min, y_min, x_max, y_max = l, t, l+w, t+h
			if digit_before_dot > 0:
				if "." or "," in text:
					output_text = "".join([char for char in text if char.isdigit()])
					if len(output_text) > digit_before_dot:
						output_text = output_text[:digit_before_dot] + "." + output_text[digit_before_dot:]
						text = output_text

			draw_boxes.append([l, t, w, h])
			draw_texts.append(text)

		draw_ids = np.argsort(np.array(draw_boxes)[:, 0])
		existed_text_boxes = []

		for idx in draw_ids:
			l, t, w, h = draw_boxes[idx] 
			x_min, y_min, x_max, y_max = l, t, l+w, t+h
			cv2.rectangle(drawImg, (x_min, y_min), (x_max, y_max), (0, 255, 0), 10)

		for idx in draw_ids:
			l, t, w, h = draw_boxes[idx]
			x_min, y_min, x_max, y_max = l, t, l+w, t+h
			text = draw_texts[idx]
			(text_width, text_height), _ = cv2.getTextSize(text, font, font_size, thickness)
			# if w/h < 1.5:
			# 	drawImg = draw_vertical_text(drawImg, text, (int(x_min),int(y_min),int(y_max)), font, font_size, (0, 0, 255), thickness,)
			# else:
				# cv2.putText(img, text, (x_min, y_max+text_height), font, font_size, (0, 0, 255), thickness)
				# continue
			if len(existed_text_boxes) == 0:
				cv2.putText(drawImg, text, (x_min, y_min), font, font_size, (0, 0, 255), thickness)
				existed_text_boxes.append([x_min, y_min, x_min+text_width, y_min+text_height])
			else:
				text_boxes = torch.Tensor(existed_text_boxes)
				box = torch.Tensor([[x_min, y_min, x_min+text_width, y_min+text_height]])
				ious = box_iou(box, text_boxes)
				if ious.max() > 0:

					cv2.putText(drawImg, text, (x_min, y_max+text_height), font, font_size, (0, 0, 255), thickness)
					existed_text_boxes.append([x_min, y_max+text_height, x_min+text_width, y_max+text_height+text_height])
				else:
					cv2.putText(drawImg, text, (x_min, y_min), font, font_size, (0, 0, 255), thickness)
					existed_text_boxes.append([x_min, y_min, x_min+text_width, y_min+text_height])
			
			

			# cv2.rectangle(drawImg, (int(x_min),int(y_min)), (int(x_max),int(y_max)), (0, 255, 0), 10)

			# # backup
			# # cv2.putText(drawImg, text, (int((x_max+x_min)/2),int(y_min)), font, font_size, (0, 0, 255), thickness,
			# # 			bottomLeftOrigin=False)
			
			# if w/h < 1.5:
			# 	drawImg = draw_vertical_text(drawImg, text, (int(x_min),int(y_min),int(y_max)), font, font_size, (0, 0, 255), thickness,)
			# else:
			# 	drawImg =  draw_horizonal_text(drawImg, text, (int(x_min),int(y_min),int(x_max)), font, font_size, (0, 0, 255), thickness)
			# cv2.putText()
		
		drawImg = cv2.resize(drawImg, (0,0), fx=0.5, fy=0.5)
		drawImg = cv2.cvtColor(drawImg, cv2.COLOR_BGR2RGB)
		pil_image = Image.fromarray(drawImg)
		bytes_image = io.BytesIO()
		pil_image.save(bytes_image, format='PNG')

		return bytes_image.getvalue()