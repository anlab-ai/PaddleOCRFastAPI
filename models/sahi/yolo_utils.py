import copy
import numpy as np
import torch
import cv2
import os

def gether_list_polys(results):
	return [p['points'] for p in results]

def process_in_batches_yolo(images_preprocessed, 
					   model, 
					   merger,
					   locations,
					   batch_size=40,
					   conf=0.3,
        			   iou=0.8,
                       device = None):
	"""
	Process images in smaller batches for inference and merging.
	
	Args:
		images_preprocessed (list): List of preprocessed images.
		shape_list_preprocessed (list): List of shape info for each image.
		model: Paddle model for inference.
		post_process_class: Post-processing class/function for predictions.
		merger: Instance of Merger class to merge polygons.
		locations: List of slice locations corresponding to polygons.
		batch_size (int): Number of images to process per batch (default: 4).
	
	Returns:
		list: Final merged polygons.
	"""
	num_images = len(images_preprocessed)
	all_phuoc_post_result = []
	all_confidences = []
	# Process images in batches
	
	for start_idx in range(0, num_images, batch_size):
		end_idx = min(start_idx + batch_size, num_images)
		batch_images = images_preprocessed[start_idx:end_idx]
		# Stack into NumPy arrays for the current batch
		# Run inference
		with torch.no_grad():
			results = model(batch_images, conf=conf, iou=iou, device=device, verbose=False)

		confidences = [result.obb.conf.cpu().numpy() for result in results]
		post_result = [result.obb.xyxyxyxy.cpu().numpy() for result in results]
		# for confidence in confidences:
		# 	confidence = confidence.tolist()
		# 	all_confidences.extend(confidence)
		# for result in post_result:
		# 	result = result.tolist()
		# 	all_phuoc_post_result.extend(result)
		all_phuoc_post_result.extend(post_result)
		all_confidences.extend(confidences)

	final_box, final_confidences = merger(polygons=copy.deepcopy(all_phuoc_post_result), slice_locations=locations, confidences=confidences)

	# return final_box, final_confidences
	return final_box, final_confidences

def draw_polygons(dt_boxes, img,color = (255, 255, 0)):


	for box in dt_boxes:
		box = np.array(box).astype(np.int32).reshape((-1, 1, 2))
		cv2.polylines(img, [box], True, color, thickness=2)
	return img