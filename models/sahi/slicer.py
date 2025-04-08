

from PIL import Image
import numpy as np
from shapely import Polygon
# Function to load image and labels
def load_image_and_labels(image_path, labels):
	"""Load image and return it with its dimensions and labels."""
	image = Image.open(image_path)
	width, height = image.size
	return image, width, height, labels

# Function to calculate slice parameters
def get_slice_parameters(width, height, slice_width=256, slice_height=256, overlap=0.2):
	"""Calculate slice coordinates based on dimensions and overlap."""
	step_x = int(slice_width * (1 - overlap))
	step_y = int(slice_height * (1 - overlap))
	x_coords = list(range(0, width, step_x))
	y_coords = list(range(0, height, step_y))
	return x_coords, y_coords, slice_width, slice_height

# Function to slice image
def slice_image(image, x, y, slice_width, slice_height, width, height):
	"""Crop a slice from the image at (x, y)."""
	min_x, min_y = x, y
	max_x = min(x + slice_width, width)
	max_y = min(y + slice_height, height)
	return image[min_y:max_y,min_x:max_x], (min_x, min_y, max_x, max_y)

# Function to check if polygon intersects slice
def polygon_intersects_slice(corners, slice_bounds):
	"""Check if polygon intersects with slice bounds."""
	min_x, min_y, max_x, max_y = slice_bounds
	x_coords, y_coords = corners[:, 0], corners[:, 1]
	return (np.max(x_coords) > min_x and np.min(x_coords) < max_x and
			np.max(y_coords) > min_y and np.min(y_coords) < max_y)
def is_poly_outside_rect(poly, x, y, w, h):
	poly = np.array(poly)
	if poly[:, 0].max() < x or poly[:, 0].min() > x + w:
		return True
	if poly[:, 1].max() < y or poly[:, 1].min() > y + h:
		return True
	return False
# Function to clip polygon to slice (simplified)
def clip_polygon_to_slice(corners, slice_bounds):
	"""Clip a clockwise polygon to slice bounds using Sutherland-Hodgman algorithm."""
	min_x, min_y, max_x, max_y = slice_bounds
	input_poly = corners.tolist()  # Input as list of [x, y] points
	
	# Define clip rectangle edges (left, right, top, bottom)
	clip_edges = [
		((min_x, min_y), (min_x, max_y)),  # Left
		((max_x, min_y), (max_x, max_y)),  # Right
		((min_x, max_y), (max_x, max_y)),  # Top
		((min_x, min_y), (max_x, min_y))   # Bottom
	]

	# Helper function to check if a point is inside an edge
	def is_inside(p, edge):
		(x1, y1), (x2, y2) = edge
		if x1 == x2:  # Vertical edge (left or right)
			return p[0] >= x1 if x1 == min_x else p[0] <= x1
		else:  # Horizontal edge (top or bottom)
			return p[1] >= y1 if y1 == min_y else p[1] <= y1

	# Helper function to compute intersection of a line segment with a clip edge
	def intersect(p1, p2, edge):
		(x1, y1), (x2, y2) = edge
		x3, y3 = p1
		x4, y4 = p2
		denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
		if denom == 0:  # Parallel lines
			return None
		t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
		u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
		if 0 <= t <= 1 and 0 <= u <= 1:
			ix = x1 + t * (x2 - x1)
			iy = y1 + t * (y2 - y1)
			return [ix, iy]
		return None

	# Sutherland-Hodgman clipping
	output_poly = input_poly
	for edge in clip_edges:
		input_poly = output_poly
		output_poly = []
		for i in range(len(input_poly)):
			p1 = input_poly[i]
			p2 = input_poly[(i + 1) % len(input_poly)]
			if is_inside(p2, edge):  # p2 inside
				if not is_inside(p1, edge):  # p1 outside
					inter = intersect(p1, p2, edge)
					if inter:
						output_poly.append(inter)
				output_poly.append(p2)
			elif is_inside(p1, edge):  # p1 inside, p2 outside
				inter = intersect(p1, p2, edge)
				if inter:
					output_poly.append(inter)

		if not output_poly:
			return None  # No intersection after clipping

	# Convert to numpy array
	clipped = np.array(output_poly)
	if len(clipped) < 3:
		return None  # Not a valid polygon

	# Ensure clockwise order
	cx, cy = np.mean(clipped[:, 0]), np.mean(clipped[:, 1])
	angles = np.arctan2(clipped[:, 1] - cy, clipped[:, 0] - cx)
	order = np.argsort(angles)
	clipped = clipped[order]
	
	# Clip coordinates to bounds
	# clipped[:, 0] = np.clip(clipped[:, 0], min_x, max_x)
	# clipped[:, 1] = np.clip(clipped[:, 1], min_y, max_y)

	return clipped

# Function to adjust labels for a slice
def adjust_labels_for_slice(labels, slice_bounds,height,width):
	"""Adjust polygon labels for a given slice."""
	min_x, min_y, max_x,  max_y = slice_bounds
	adjusted_labels = []
	
	for label in labels:
		
		corners = np.array(label).reshape(4, 2)  # [x1, y1], [x2, y2], ...
	   
		if not is_poly_outside_rect( corners, min_x,min_y,max_x-min_x,max_y-min_y):
		   
			clipped_corners = clip_polygon_to_slice(corners, slice_bounds)
			
			if clipped_corners is not None:
				# Shift to slice-local coordinates
				clipped_corners -= [min_x, min_y]
				adjusted_label =  clipped_corners.flatten().tolist()
				adjusted_labels.append(adjusted_label)
	
	return adjusted_labels

# Main function to process everything
def process_image_and_labels(image, labels,**kwargs):
	"""Process image slicing and label adjustment."""
	# Load image and labels
	if image is None:
		return [], [],[]
	height,width = image.shape[0], image.shape[1]
	# Get slice parameters
	x_coords, y_coords, slice_width, slice_height = get_slice_parameters(width, height,**kwargs)
	
	# Store results
	slices = []
	slice_labels = []
	all_slice_bounds = []
	# Slice and adjust labels
	for y in y_coords:
		for x in x_coords:
			slice_img, slice_bounds = slice_image(image, x, y, slice_width, slice_height, width, height)
		
			adjusted_labels = adjust_labels_for_slice(labels, slice_bounds,height,width)
			
			# if len(adjusted_labels) == 0:
			#     continue
			slices.append(np.copy(slice_img))
			slice_labels.append(adjusted_labels)
			all_slice_bounds.append(slice_bounds)
			# slice_img.save(f"slice_{x}_{y}.jpg")
	
	return slices, slice_labels,all_slice_bounds
def clip_polygon_to_slice_v2(corners, slice_bounds):
	"""Clip a clockwise polygon to slice bounds using Sutherland-Hodgman algorithm."""
	min_x, min_y, max_x, max_y = slice_bounds
	region_polys = np.array(
		[
			(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)
		]
	)
	
	region_polys = Polygon(region_polys)
	input_poly = Polygon(corners)
	clipped = region_polys.intersection(input_poly)
	if clipped.geom_type == 'Polygon':
	# Only access 'exterior' if the intersection is a Polygon
		
		if clipped.area < 0.0003*((max_x-min_x)*(max_y-min_y))*(840/(max_y-min_y)):
			return None
		clipped = clipped.exterior.coords.xy
	elif clipped.geom_type == 'Point':
		return None
	else:
		print(f"Unexpected intersection type: {clipped.geom_type}")
		return None
  
	clipped = np.array(list(zip(*clipped)))
	if len(clipped) == 0:
		return None  # No intersection after clipping
	try:
		cx, cy = np.mean(clipped[:, 0]), np.mean(clipped[:, 1])
	except:
		import pdb;pdb.set_trace()
		

	angles = np.arctan2(clipped[:, 1] - cy, clipped[:, 0] - cx)
	order = np.argsort(angles)
	clipped = clipped[order]

	return clipped
def adjust_labels_for_slice_v2(labels, slice_bounds):
	"""Adjust polygon labels for a given slice."""
	min_x, min_y, max_x,  max_y = slice_bounds
	adjusted_labels = []
	
	for label in labels:
		
		corners = np.array(label).reshape(4, 2)  # [x1, y1], [x2, y2], ...
	   
		if not is_poly_outside_rect( corners, min_x,min_y,max_x-min_x,max_y-min_y):
		   
			clipped_corners = clip_polygon_to_slice_v2(corners, slice_bounds)
			
			if clipped_corners is not None:
				# Shift to slice-local coordinates
				clipped_corners -= [min_x, min_y]
				adjusted_label =  clipped_corners.flatten().tolist()
				adjusted_labels.append(adjusted_label)
	
	return adjusted_labels
def process_image_and_labels_v2(image, labels,**kwargs):
	"""Process image slicing and label adjustment."""
	# Load image and labels
	if image is None:
		return [], [],[]
	height,width = image.shape[0], image.shape[1]
	# Get slice parameters
	x_coords, y_coords, slice_width, slice_height = get_slice_parameters(width, height,**kwargs)
	
	# Store results
	slices = []
	slice_labels = []
	all_slice_bounds = []
	# Slice and adjust labels
	for y in y_coords:
		for x in x_coords:
			slice_img, slice_bounds = slice_image(image, x, y, slice_width, slice_height, width, height)
		
			adjusted_labels = adjust_labels_for_slice_v2(labels, slice_bounds)
			
			slices.append(np.copy(slice_img))
			slice_labels.append(adjusted_labels)
			all_slice_bounds.append(slice_bounds)
	
	return slices, slice_labels,all_slice_bounds

if __name__ ==  '__main__':
	# Example usage
	image_path = "path/to/your/image.jpg"
	labels = [
		[0, 150, 200, 250, 200, 250, 300, 150, 300],  # Class 0, rotated rectangle
		[1, 450, 550, 600, 550, 600, 650, 450, 650],  # Class 1, another rotated box
	]

	slices, slice_labels = process_image_and_labels(image_path, labels)

	# Print results
	for i, (slice_img, labels_in_slice) in enumerate(zip(slices, slice_labels)):
		print(f"Slice {i} ({slice_img.size}):")
		for label in labels_in_slice:
			print(f"  Polygon Label: {label}")

class SliceImage:
	def __init__(self,
				 slice_height, 
				 slice_width,
	 			 overlap,
      			 **kwargs):
		self.slice_height = slice_height
		self.slice_width = slice_width
		self.overlap = overlap
	def __call__(self, image):
		height,width = image.shape[0], image.shape[1]
		x_coords, y_coords, slice_width, slice_height = get_slice_parameters(width, height, 
																	   slice_width=self.slice_width,
																	   slice_height=self.slice_height, 
																	   overlap=self.overlap)
		# Store results
		slices = []
		all_slice_bounds = []
		# Slice and adjust labels
		for y in y_coords:
			for x in x_coords:
				slice_img, slice_bounds = slice_image(image, 
                                          				x,
                                              			y,
                                                 		slice_width,
                                                   		slice_height,
                                                     	width,
                                                      	height)
				slices.append(np.copy(slice_img))
				all_slice_bounds.append(slice_bounds)
			
		return slices,all_slice_bounds