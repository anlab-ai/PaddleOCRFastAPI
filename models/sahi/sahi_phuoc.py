from ultralytics import YOLO
import cv2
from .slicer import SliceImage
from .merger import Merger
from .yolo_utils import process_in_batches_yolo,draw_polygons
import os
import numpy as np

class YOLO_SAHI:
	def __init__(self,
			  	model_path,
				confidence_threshold,
				device="cuda:0",
				**kwargs
			  ):
		self.core_detection = YOLO(model_path)
		# self.core_detection.eval()
		
		self.device = device
		self.confidence_threshold = confidence_threshold
		self.kwrags = kwargs
		self.img_slicer = SliceImage(**kwargs)
		self.sahi_merge  = Merger(**kwargs)
		
	def predict_from_path(self, img):
		if isinstance(img, str):
			original_image = cv2.imread(img)
		elif isinstance(img, np.ndarray):
			original_image = img.copy()

		sliced_images, all_slice_bounds = self.img_slicer(original_image)
		print(f"Len slice image: {len(sliced_images)}")
		# print(len(sliced_images))
		# print(all_slice_bounds)
		# for idx, image in enumerate(sliced_images):
		# 	results = self.core_detection.predict(image)
		# 	results[0].save(f"/media/hieu/data/download/Archive/2/{idx}.jpg")
			# cv2.imwrite(f"/media/hieu/data/download/Archive/2/{idx}.jpg", image)
		from time import time
		t1 = time()
		sahi_box,sahi_confidences = process_in_batches_yolo(sliced_images,
														self.core_detection,
														self.sahi_merge,
														all_slice_bounds,
														batch_size=1,
														conf=self.confidence_threshold,
														iou=0.7,
														device = self.device)
		print("Time: ",time()-t1)
		if len(sahi_box) == 0:
			return [], []
		# print(f"Sahi box: {len(sahi_box)}")
		# print("Process yolo done!")
		image_size = (original_image.shape[1],original_image.shape[0])
		dbscan_final_box,dbscan_final_confidences = self.sahi_merge.dbcan_merge(sahi_box,
                                                                      		image_size = image_size,
                                                                        	final_confidences = sahi_confidences)
		return dbscan_final_box,dbscan_final_confidences

	def visualize(self,
               img_path,
               save_folder,
               box_result):
     
		os.makedirs(save_folder,exist_ok=True)
		img_id = os.path.basename(img_path)
		save_path = os.path.join(save_folder,img_id)
		img = cv2.imread(img_path)
		img = draw_polygons(box_result,img)
		cv2.imwrite(save_path,img)