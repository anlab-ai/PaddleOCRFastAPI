import os
import numpy as np
import cv2
import math

def clockwise_sort(points):
    points = points.squeeze()
    # Compute centroid (average of all points)
    cx = sum(x for x, y in points) / len(points)
    cy = sum(y for x, y in points) / len(points)

    # Sort points by angle with respect to the centroid
    points = sorted(points,key=lambda p: math.atan2(p[1] - cy, p[0] - cx), reverse=True)
    return points
def draw_det_res(dt_boxes, img, img_name, save_path,color = (255, 255, 0)):

	src_im = img
	for box in dt_boxes:
		box = np.array(box).astype(np.int32).reshape((-1, 1, 2))
		cv2.polylines(src_im, [box], True, color, thickness=2)
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	save_path = os.path.join(save_path, os.path.basename(img_name))
	cv2.imwrite(save_path, src_im)
 
def get_max_width_height(rect):
	"""Tính chiều rộng và chiều cao tối đa của hình chữ nhật"""
	widthA = np.linalg.norm(rect[2] - rect[3])  # BR - BL
	widthB = np.linalg.norm(rect[1] - rect[0])  # TR - TL
	maxWidth = int(max(widthA, widthB))

	heightA = np.linalg.norm(rect[1] - rect[2])  # TR - BR
	heightB = np.linalg.norm(rect[0] - rect[3])  # TL - BL
	maxHeight = int(max(heightA, heightB))

	return maxWidth, maxHeight
def order_points(pts):
	
	"""Sắp xếp 4 điểm theo thứ tự: [TL, TR, BR, BL]"""
	pts = pts[np.argsort(pts[:, 0])]  # Sắp xếp theo tọa độ x
	left_pts, right_pts = pts[:2], pts[2:]
	
	# Xác định top-left và bottom-left theo y
	left_pts = left_pts[np.argsort(left_pts[:, 1])]
	right_pts = right_pts[np.argsort(right_pts[:, 1])]
	
	rect = np.array([left_pts[0], right_pts[0], right_pts[1], left_pts[1]], dtype=np.float32)
	return rect

def crop_region(image, rect):
    """Cắt ảnh theo bounding box của polygon và cập nhật tọa độ điểm"""
    # Lấy tọa độ min/max từ rect
    x_min, y_min = np.min(rect, axis=0).astype(int)
    x_max, y_max = np.max(rect, axis=0).astype(int)
    
    # Đảm bảo tọa độ nằm trong giới hạn ảnh
    height, width = image.shape[:2]
    x_min = max(0, x_min)  # Không nhỏ hơn 0
    y_min = max(0, y_min)  # Không nhỏ hơn 0
    x_max = min(width, x_max)  # Không vượt quá width
    y_max = min(height, y_max)  # Không vượt quá height
    
    # Kiểm tra nếu vùng cắt hợp lệ
    if x_max <= x_min or y_max <= y_min:
        # Trả về ảnh rỗng và rect gốc nếu vùng không hợp lệ
        return np.zeros((0, 0, image.shape[2] if len(image.shape) > 2 else 0), dtype=image.dtype), rect
    
    # Cắt ảnh
    cropped = image[y_min:y_max, x_min:x_max].copy()
    
    # Cập nhật lại tọa độ điểm cho ảnh đã cắt
    new_rect = rect - np.array([x_min, y_min]).reshape(1, 2)
    
    # Đảm bảo tọa độ mới không âm
    new_rect = np.clip(new_rect, 0, None).astype(np.float32)
    
    return cropped, new_rect

def perspective_transform(image, pts):
	"""
	Cắt và biến đổi phối cảnh một vùng trong ảnh thành hình chữ nhật.
	Args:
		image: Input image (numpy array)
		pts: Array of points defining the region to transform
	Returns:
		warped: Transformed rectangular image
	"""
	try:
		# Check if input is valid
		if image is None or pts is None:
			raise ValueError("Image or points cannot be None")
		
		if len(pts) != 4:
			raise ValueError("Exactly 4 points are required for perspective transform")
			
		# Convert points to numpy array if not already
		polygons = np.array(pts, dtype=np.float32)
		
		maxWidth, maxHeight = get_max_width_height(polygons)
	
		# Define destination points
		dst = np.array([
			[0, 0],
			[maxWidth - 1, 0],
			[maxWidth - 1, maxHeight - 1],
			[0, maxHeight - 1]
		], dtype=np.float32)
		
		# Calculate perspective transform matrix
		M = cv2.getPerspectiveTransform(polygons, dst)
		
		# Check if transformation matrix is valid
		if M is None:
			raise ValueError("Failed to compute perspective transform matrix")
			
		# Apply perspective transform with border handling
		warped = cv2.warpPerspective(
			image, 
			M, 
			(maxWidth, maxHeight),
			flags=cv2.INTER_LINEAR,
			borderMode=cv2.BORDER_CONSTANT,
			borderValue=(0, 0, 0)  # Black border for out-of-bounds areas
		)
		
		return warped
		
	except Exception as e:
		print(f"Error in perspective_transform: {str(e)}")
		return None
	
	