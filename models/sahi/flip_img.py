import numpy as np
import cv2

def flip_vertical(img):
	h,w,_ = img.shape
	src = np.array([
				[0, 0],
				[w - 1, 0],
				[w - 1, h - 1],
				[0, h - 1]
			], dtype=np.float32)
 
	dst = np.array([
				[0, h - 1],
				[w - 1, h - 1],
				[w - 1, 0],
				[0, 0],
			], dtype=np.float32)
	M = cv2.getPerspectiveTransform(src, dst)

	# Check if transformation matrix is valid
	if M is None:
		raise ValueError("Failed to compute perspective transform matrix")
		
	# Apply perspective transform with border handling
	warped = cv2.warpPerspective(
		img, 
		M, 
		(w, h),
		flags=cv2.INTER_LINEAR,
		borderMode=cv2.BORDER_CONSTANT,
		borderValue=(0, 0, 0)  # Black border for out-of-bounds areas
	)
	return warped
def flip_left(img):
	h,w,_ = img.shape
	src = np.array([
				  	[0, 0],        # top-left
					[w - 1, 0],    # top-right
					[w - 1, h - 1], # bottom-right
					[0, h - 1] 
			], dtype=np.float32)
 
	dst = np.array([
				[0, w-1],
				[0,0],
				[h-1,0],
				[h-1,w-1],
				
			], dtype=np.float32)
	M = cv2.getPerspectiveTransform(src, dst)

	# Check if transformation matrix is valid
	if M is None:
		raise ValueError("Failed to compute perspective transform matrix")
		
	# Apply perspective transform with border handling
	warped = cv2.warpPerspective(
		img, 
		M, 
		(h, w),
		flags=cv2.INTER_LINEAR,
		borderMode=cv2.BORDER_CONSTANT,
		borderValue=(0, 0, 0)  # Black border for out-of-bounds areas
	)
	return warped
def flip_right(img):

   
	# Get original dimensions
	h, w, _ = img.shape
	
	# Source points (original corners)
	src = np.array([
		[0, 0],        # top-left
		[w - 1, 0],    # top-right
		[w - 1, h - 1], # bottom-right
		[0, h - 1]      # bottom-left
	], dtype=np.float32)
	
	# Destination points for 90° clockwise rotation
	# Mapped to new dimensions (h, w)
	dst = np.array([
		[h - 1, 0],    # new top-right
		[h - 1, w - 1], # new bottom-right
		[0, w - 1],     # new bottom-left
		[0, 0]          # new top-left
	], dtype=np.float32)
	
	# Compute transformation matrix
	M = cv2.getPerspectiveTransform(src, dst)
 
	# Apply transform with new dimensions
	rotated = cv2.warpPerspective(
		img,
		M,
		(h, w),  # New width = old height, new height = old width
		flags=cv2.INTER_LINEAR,
		borderMode=cv2.BORDER_CONSTANT,
		borderValue=(0, 0, 0)
	)
	return rotated
	
class Rotator:
	def __init__(self,yolo_path,img_size = 640):
		from ultralytics import YOLO
		self.model = YOLO(yolo_path)
		self.img_size = img_size
	def get_rotate_images(self,original_image,mean_points,x_min_ratio = 0.4,x_max_ratio = 0.6,y_line_ratio = 0.4):
		
		x_mean, y_mean = mean_points  # Corrected typo from mean_poits
  
		if y_mean < y_line_ratio and ( x_mean > x_max_ratio or x_mean < x_min_ratio):
				final_top1 = '270' if x_mean > x_max_ratio else '90'
				conf = 1
		else:
		
			# Run the model once
			predict = self.model(original_image,verbose = False, imgsz=self.img_size)
			top1 = predict[0].probs.top1
			conf = predict[0].probs.top1conf.item()
			class_name = self.model.names[top1]

			# Adjust prediction based on x_mean for '90' and '270'
			if class_name == '90':
				final_top1 = '270' if x_mean > x_max_ratio else '90'
			elif class_name == '270':
				final_top1 = '90' if x_mean < x_min_ratio else '270'
			else:
				final_top1 = class_name  # '0' or other classes

		# Apply the appropriate transformation
		if final_top1 == '0':
			image_list = [original_image]
		elif final_top1 == '90':
			image_list = [flip_left(original_image)]  # Rotate 90° counterclockwise
		elif final_top1 == '270':
			image_list = [flip_right(original_image)]  # Rotate 90° clockwise
		else:
			image_list = [original_image]  # Default for unhandled cases (e.g., '180')

		# Prepare return values
		top1_list = [final_top1]
		conf_list = [conf]

		# Return corrected image, confidence, and prediction
		return image_list, [round(conf * 100, 2)], top1_list


			
	def get_rotate_images_backup(self,original_image):
		h,w,_ = original_image.shape
		
		rotated_images = [
			original_image,
			flip_vertical(np.copy(original_image)),
			flip_left(np.copy(original_image)),
			flip_right(np.copy(original_image)),
		]
		return rotated_images
			
	
if __name__ == "__main__":
	img = cv2.imread("/work/21013187/phuoc/paddle_detect/data/anh_lem_mau.jpg")
	cv2.imwrite("origin.png", img)
	flipped_img = flip_vertical(img)
	cv2.imwrite("test.png", flipped_img)
	flipped_img = flip_left(img)
	cv2.imwrite("test_left.png", flipped_img)
	flipped_img = flip_right(img)
	cv2.imwrite("test_right.png", flipped_img)