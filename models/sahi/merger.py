import os
import cv2
import math
import numpy as np
from itertools import combinations
import random
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .dbscan import DBScanPhuoc

def clockwise_sort(points):
    points = points.squeeze()
    # Compute centroid (average of all points)
    cx = sum(x for x, y in points) / len(points)
    cy = sum(y for x, y in points) / len(points)

    # Sort points by angle with respect to the centroid
    points = sorted(points,key=lambda p: math.atan2(p[1] - cy, p[0] - cx), reverse=True)
    return points

def compute_iou(poly1, poly2):
	# import pdb;pdb.set_trace()
	# Convert to Shapely polygons (closed loop: repeat first point)
	
	p1 = Polygon(poly1.squeeze())
	p2 = Polygon(poly2.squeeze())
	union = p1.union(p2).area
	if union <= 0:
			return 0
	intersection = p1.intersection(p2).area
	try:
		return max(intersection/p1.area,intersection/p2.area)
	except:
		return 0

	

class Merger:
	def __init__(self, iou_merge_sahi=0.55,**kwargs):
		self.iou_thres = iou_merge_sahi
		self.db_scan = DBScanPhuoc(**kwargs)
	def __call__(self, polygons, slice_locations,confidences):
		
		assert len(polygons) == len(slice_locations) , print(len(polygons),len(slice_locations),sep='\n')

		self.all_polys = []
		self.poly_map = {}
		self.all_confidences = []
	
		# Process each polygon's slice and adjust coordinates
		for i_frame,(polys, location,confidence) in enumerate(zip(polygons, slice_locations,confidences)):
			x1, y1, x2, y2 = location
			for conf in confidence:
				self.all_confidences.append(float(conf))
    
			for poly in polys:
				self.poly_map[len(self.all_polys)] = i_frame
				# Ensure poly is a flat list or 2D array, then reshape to (n, 2)
				poly = np.array(poly).reshape(-1, 2)  # Convert to (n_points, 2)
				# Adjust coordinates by slice location
				poly = poly + np.array([x1, y1])
				self.all_polys.append(poly)
		# Merge polygons based on IoU
		new_polys,conf = self.merge()
		
		assert(len(new_polys) == len(conf))
		return new_polys,conf

	# def __call__(self, polygons, slice_locations):
		
	# 	assert len(polygons) == len(slice_locations) , print(len(polygons),len(slice_locations),sep='\n')

	# 	self.all_polys = []
	# 	self.poly_map = {}
	# 	# Process each polygon's slice and adjust coordinates
	# 	for i_frame,(polys, location) in enumerate(zip(polygons, slice_locations)):
	# 		x1, y1, x2, y2 = location
			
	# 		for poly in polys:
	# 			self.poly_map[len(self.all_polys)] = i_frame
	# 			# Ensure poly is a flat list or 2D array, then reshape to (n, 2)
	# 			poly = np.array(poly).reshape(-1, 2)  # Convert to (n_points, 2)
	# 			# Adjust coordinates by slice location
	# 			poly = poly + np.array([x1, y1])
	# 			self.all_polys.append(poly)
	# 	# Merge polygons based on IoU
	# 	new_polys = self.merge()
	# 	return new_polys
	def visualize(self,image,polys,color = None):
		for poly in polys:
			poly = np.array(poly).reshape(-1, 2)  # Ensure (n, 2) shape
			poly_int = poly.astype(np.int32).reshape(-1, 1, 2)  # For cv2.polylines
			color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) if color is  None else color
			cv2.polylines(image, [poly_int], isClosed=True, color=color, thickness=2)
		return image

	def merge(self):
		polys = self.all_polys
		# Initialize each polygon as its own group
		groups = [[i] for i in range(len(polys))]

		# Compute IoU for all pairs and group overlapping polygons
		for (i, j) in combinations(range(len(polys)), 2):
			if self.poly_map[i] == self.poly_map[j]:
				continue
			poly_i = polys[i]
			poly_j = polys[j]
			iou = compute_iou(poly_i, poly_j)
			
			if iou > self.iou_thres:
	   
				# Find and merge groups containing i or j
				for idx, g in enumerate(groups):
					if i in g or j in g:
						groups[idx] = list(set(g + [i, j]))  # Merge and deduplicate
						break

		# Remove duplicate groups
		final_groups = []
		seen = set()
		for g in groups:
			group_idx = tuple(sorted(g))
			
			for i in group_idx:
				if not i in seen:
					is_seen = False
				else:
					is_seen = True
			if not is_seen:
				seen = seen.union(group_idx) 
				final_groups.append(g)

		# Create merged polygons for each group
		result = []
		confidences = []
		for g in final_groups:
			if len(g) == 1:
	
				result.append(polys[g[0]])  # Keep as (n, 2) array
				confidences.append(self.all_confidences[g[0]])
				
			else:
				confidences.append(np.mean([self.all_confidences[i] for i in g]))
				# confidences.append(self.all_confidences[g[0]])
				merged_poly = self.merge_polygons([polys[i] for i in g])
				result.append(merged_poly)

		return result, confidences
	# def merge(self):
	# 	polys = self.all_polys
	# 	# Initialize each polygon as its own group
	# 	groups = [[i] for i in range(len(polys))]

	# 	# Compute IoU for all pairs and group overlapping polygons
	# 	for (i, j) in combinations(range(len(polys)), 2):
	# 		if self.poly_map[i] == self.poly_map[j]:
	# 			continue
	# 		poly_i = polys[i]
	# 		poly_j = polys[j]
	# 		iou = compute_iou(poly_i, poly_j)
			
	# 		if iou > self.iou_thres:
	   
	# 			# Find and merge groups containing i or j
	# 			for idx, g in enumerate(groups):
	# 				if i in g or j in g:
	# 					groups[idx] = list(set(g + [i, j]))  # Merge and deduplicate
	# 					break

	# 	# Remove duplicate groups
	# 	final_groups = []
	# 	seen = set()
	# 	for g in groups:
	# 		group_idx = tuple(sorted(g))
			
	# 		for i in group_idx:
	# 			if not i in seen:
	# 				is_seen = False
	# 			else:
	# 				is_seen = True
	# 		if not is_seen:
	# 			seen = seen.union(group_idx) 
	# 			final_groups.append(g)

	# 	# Create merged polygons for each group
	# 	result = []
	# 	for g in final_groups:
	# 		if len(g) == 1:
	# 			result.append(polys[g[0]])  # Keep as (n, 2) array
	# 		else:
	# 			merged_poly = self.merge_polygons([polys[i] for i in g])
	# 			result.append(merged_poly)

	# 	return result

	

	def merge_polygons_not_intersection(self, poly_list):
		"""Merge a list of polygons into a single bounding box."""
		try:
			all_x = np.concatenate([np.array(p)[:, 0] for p in poly_list if len(p) > 1])
		except:
			import pdb;pdb.set_trace()
		all_y = np.concatenate([np.array(p)[:, 1] for p in poly_list])
		x_min, x_max = np.min(all_x), np.max(all_x)
		y_min, y_max = np.min(all_y), np.max(all_y)
		return np.array([
			[x_min, y_min],  # Top-left
			[x_max, y_min],  # Top-right
			[x_max, y_max],  # Bottom-right
			[x_min, y_max]   # Bottom-left
		])
	def merge_polygons(self, polys):
		"""
		Merge multiple polygons into a single polygon using unary union.
		
		Args:
			polys: List of polygon coordinates, where each polygon is a list/array of (x,y) points
			
		Returns:
			numpy.ndarray: Array of merged polygon exterior coordinates as int32
			
		Raises:
			ValueError: If input is invalid or polygons cannot be merged
		"""
		try:
			# Check if input is valid
			if not polys or not isinstance(polys, (list, tuple)):
				raise ValueError("Input must be a non-empty list or tuple of polygons")
				
			# Convert input coordinates to Polygon objects
			polygon_objects = []
			for poly in polys:
				if len(poly) < 3:  # A polygon needs at least 3 points
					raise ValueError("Each polygon must have at least 3 points")
				polygon_objects.append(Polygon(poly))
				
			# Perform the union operation
			merged_poly = unary_union(polygon_objects)
			
			# Handle case where merged_poly might be a MultiPolygon
			if merged_poly.geom_type == 'MultiPolygon':
				# Take the largest polygon by area if multiple disjoint polygons result
				merged_poly = max(merged_poly.geoms, key=lambda x: x.area)
			elif merged_poly.geom_type != 'Polygon':
				raise ValueError("Resulting geometry is not a valid polygon")
				
			# Convert to numpy array with integer coordinates
			coords = np.array(merged_poly.exterior.coords, dtype=np.int32)
			
			return coords
			
		except Exception as e:
			raise ValueError(f"Error merging polygons: {str(e)}")
	def dbcan_merge(self,final_polys,image_size,final_confidences):
		self.db_scan.feetch_data(final_polys,image_size)
		self.db_scan.fit()
		labels = self.db_scan.labels
		merged_polys = []
		merged_confs = []
		for label in np.unique(labels):
			all_index = [i for i in np.where(labels == label)[0] ]
			cluster_polys = [final_polys[i] for i in all_index]
			merged_confs.append(np.mean([final_confidences[i] for i in all_index]))
			if len(cluster_polys) ==1:
				merged_poly = cluster_polys[0]

			else:
				merged_poly = self.merge_polygons_not_intersection(cluster_polys)
			merged_polys.append(merged_poly)
		assert(len(merged_polys) == len(merged_confs))
	
		return merged_polys,merged_confs
	def dbcan_merge_backup(self,final_polys,image_size ):
		self.db_scan.feetch_data(final_polys,image_size)
		self.db_scan.fit()
		labels = self.db_scan.labels
		merged_polys = []
		for label in np.unique(labels):
		
			cluster_polys = [final_polys[i] for i in np.where(labels == label)[0]]
			if len(cluster_polys) ==1:
				merged_poly = cluster_polys[0]
			else:
				merged_poly = self.merge_polygons_not_intersection(cluster_polys)
			merged_polys.append(merged_poly)
	
		return merged_polys

	def convert_to_four_point_helpers(self, coords):
		"""
		Convert a polygon with any number of points to a 4-point polygon by selecting
		extremum points and ensuring a valid quadrilateral.
		
		Args:
			coords: numpy.ndarray of shape (n, 2) with polygon coordinates, where n >= 3
			
		Returns:
			numpy.ndarray: 4-point polygon coordinates as int32
			
		Raises:
			ValueError: If input has fewer than 3 points or invalid dimensions
		"""
		# Input validation
		coords = np.array(coords).squeeze()
		if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 3:
			raise ValueError("Input must be an array of shape (n, 2) with n >= 3")
			
		# If already 4 points, return as-is
		if len(coords) == 4:
			return clockwise_sort(coords.astype(np.int32))
		
		# Handle special case: if 3 points, create rectangle by adding a fourth point
		if len(coords) == 3:
			# Calculate vectors between points
			v1 = coords[1] - coords[0]
			v2 = coords[2] - coords[1]
			
			# Create perpendicular vector for fourth point
			perp = np.array([-v1[1], v1[0]])
			fourth_point = coords[2] + perp
			four_points = np.vstack([coords, fourth_point])
			return clockwise_sort(four_points.astype(np.int32))
		
		# For > 4 points, find extremum points
		min_x_idx = np.argmin(coords[:, 0])
		max_x_idx = np.argmax(coords[:, 0])
		min_y_idx = np.argmin(coords[:, 1])
		max_y_idx = np.argmax(coords[:, 1])
		
		# Get unique extremum points (handling cases where points might coincide)
		extremum_indices = list(set([min_x_idx, max_x_idx, min_y_idx, max_y_idx]))
		
		# If we don't have 4 unique points, supplement with additional points
		if len(extremum_indices) < 4:
			return coords
		extremum_points = coords[extremum_indices]
		return clockwise_sort(extremum_points.astype(np.int32))
		
		
	def convert_to_four_points(self,polys):
		"""
		Convert polygon coordinates to 4-point polygons using cv2.approxPolyDP.
		
		Args:
			polys: List of polygon coordinates, where each polygon is a list/array of (x,y) points
			
		Returns:
			numpy.ndarray: Array of 4-point polygon coordinates as int32
		"""
		return [self.convert_to_four_point_helpers(poly) for poly in polys]
# Example usage
if __name__ == "__main__":
	# Dummy data
	image = np.zeros((500, 500, 3), dtype=np.uint8)
	polygons = [
		[[100, 100, 150, 100, 150, 150, 100, 150]],  # Slice 1: one polygon
		[[200, 200, 250, 200, 250, 250, 200, 250]]   # Slice 2: one polygon
	]
	slice_locations = [(0, 0, 100, 100), (50, 50, 150, 150)]  # Overlapping slices

	merger = Merger(iou_thres=0.1)  # Low threshold for testing overlap
	result_image = merger(image, polygons, slice_locations)

	cv2.imshow("Merged Polygons", result_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()