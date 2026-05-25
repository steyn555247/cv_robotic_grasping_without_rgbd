import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import gc
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict
import plotly.graph_objects as go
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# ==========================================
# 1. CORE CLASSES (Grasping Logic)
# ==========================================

@dataclass
class GraspCandidate:
    x: int
    y: int
    angle: float
    width: float
    height: float = 20.0
    edge_quality: float = 0.0
    depth_quality: float = 0.0
    cog_quality: float = 0.0
    combined_quality: float = 0.0
    line_length: float = 0.0  # NEW: Total line length through CoG

    def get_corners(self) -> np.ndarray:
        cos_a = np.cos(self.angle)
        sin_a = np.sin(self.angle)
        w, h = self.width, self.height
        
        corners_local = [
            (-w/2, -h/2), (w/2, -h/2),
            (w/2, h/2), (-w/2, h/2)
        ]
        
        corners_global = []
        for dx, dy in corners_local:
            px = self.x + dx * cos_a - dy * sin_a
            py = self.y + dx * sin_a + dy * cos_a
            corners_global.append([px, py])
            
        return np.array(corners_global)

class ImprovedGraspDetector:
    def __init__(self, edge_weight, depth_weight, cog_weight):
        self.edge_weight = edge_weight
        self.depth_weight = depth_weight
        self.cog_weight = cog_weight

    def _cast_ray_direct_line(self, mask, grasp_point, grad_x, grad_y, contour, gradient_source="Depth Gradients", max_dist=500):
        """
        Cast a ray FROM grasp_point in perpendicular direction using various gradient methods.
        Does NOT necessarily go through CoG.
        
        Args:
            mask: Binary mask where 255=object, 0=background
            grasp_point: (x, y) tuple on the contour
            grad_x: X-component of gradient map
            grad_y: Y-component of gradient map
            contour: The object contour (for contour-based methods)
            gradient_source: Method to determine direction
            max_dist: Maximum distance to search
            
        Returns:
            (end1_x, end1_y, end2_x, end2_y, total_length)
        """
        h, w = mask.shape
        gx, gy = grasp_point
        
        if gradient_source == "Contour Direction (80px avg)":
            # Find grasp point on contour
            if contour is not None:
                contour_points = contour.reshape(-1, 2)
                
                # Find closest contour point to grasp point
                distances = np.sqrt((contour_points[:, 0] - gx)**2 + (contour_points[:, 1] - gy)**2)
                closest_idx = np.argmin(distances)
                
                # Calculate how many points = ~80px
                # Assuming contour points are roughly 1-2 pixels apart
                num_points = min(40, len(contour_points) // 4)  # 40 points * ~2px/point ≈ 80px
                
                # Get segment around grasp point
                half_segment = num_points // 2
                
                # Handle wraparound
                start_idx = (closest_idx - half_segment) % len(contour_points)
                end_idx = (closest_idx + half_segment) % len(contour_points)
                
                if start_idx < end_idx:
                    segment = contour_points[start_idx:end_idx+1]
                else:
                    # Wraparound case
                    segment = np.vstack([contour_points[start_idx:], contour_points[:end_idx+1]])
                
                # Calculate tangent direction by averaging differences
                if len(segment) > 2:
                    # Method 1: Linear regression for more robust direction
                    # Fit line through segment points
                    x_coords = segment[:, 0]
                    y_coords = segment[:, 1]
                    
                    # Center the coordinates
                    x_mean = np.mean(x_coords)
                    y_mean = np.mean(y_coords)
                    x_centered = x_coords - x_mean
                    y_centered = y_coords - y_mean
                    
                    # Calculate covariance
                    cov_xx = np.sum(x_centered * x_centered)
                    cov_xy = np.sum(x_centered * y_centered)
                    cov_yy = np.sum(y_centered * y_centered)
                    
                    # Eigenvector of covariance matrix gives principal direction
                    # Tangent direction is the principal component
                    if cov_xx + cov_yy > 1e-6:
                        # Calculate eigenvalues and eigenvectors analytically for 2x2
                        trace = cov_xx + cov_yy
                        det = cov_xx * cov_yy - cov_xy * cov_xy
                        lambda1 = trace/2 + np.sqrt(max(0, (trace/2)**2 - det))
                        
                        # Eigenvector for largest eigenvalue
                        if abs(cov_xy) > 1e-6:
                            tx = lambda1 - cov_yy
                            ty = cov_xy
                        elif abs(cov_xx - lambda1) > 1e-6:
                            tx = cov_xy
                            ty = lambda1 - cov_xx
                        else:
                            # Use simple difference
                            tx = segment[-1, 0] - segment[0, 0]
                            ty = segment[-1, 1] - segment[0, 1]
                        
                        # Normalize tangent
                        t_length = np.sqrt(tx**2 + ty**2)
                        if t_length > 1e-6:
                            tx /= t_length
                            ty /= t_length
                            
                            # Perpendicular: rotate tangent 90 degrees
                            dx = -ty
                            dy = tx
                        else:
                            # Fallback
                            dx, dy = 1.0, 0.0
                    else:
                        # Fallback
                        dx, dy = 1.0, 0.0
                else:
                    # Not enough points, use simple difference
                    if len(segment) >= 2:
                        tx = segment[-1, 0] - segment[0, 0]
                        ty = segment[-1, 1] - segment[0, 1]
                        t_length = np.sqrt(tx**2 + ty**2)
                        if t_length > 1e-6:
                            tx /= t_length
                            ty /= t_length
                            dx = -ty
                            dy = tx
                        else:
                            dx, dy = 1.0, 0.0
                    else:
                        dx, dy = 1.0, 0.0
            else:
                # No contour, fallback to radial
                dx = gx - w // 2
                dy = gy - h // 2
                length = np.sqrt(dx**2 + dy**2)
                if length > 1e-6:
                    dx /= length
                    dy /= length
                else:
                    dx, dy = 1.0, 0.0
        
        elif gradient_source == "Depth Gradients":
            # Use depth gradient at this point (most reliable for depth-based grasping)
            if 0 <= gy < grad_y.shape[0] and 0 <= gx < grad_x.shape[1]:
                gx_val = grad_x[gy, gx]
                gy_val = grad_y[gy, gx]
            else:
                # Fallback to sampling around the point
                kernel_size = 3
                y_min = max(0, gy - kernel_size)
                y_max = min(h, gy + kernel_size + 1)
                x_min = max(0, gx - kernel_size)
                x_max = min(w, gx + kernel_size + 1)
                
                gx_val = np.mean(grad_x[y_min:y_max, x_min:x_max])
                gy_val = np.mean(grad_y[y_min:y_max, x_min:x_max])
            
            # Calculate perpendicular direction
            length = np.sqrt(gx_val**2 + gy_val**2)
            if length < 1e-6:
                # Fallback to radial
                dx = gx - w // 2
                dy = gy - h // 2
                length = np.sqrt(dx**2 + dy**2)
                if length < 1e-6:
                    dx, dy = 1.0, 0.0
                else:
                    dx /= length
                    dy /= length
            else:
                # Perpendicular to gradient: swap and negate one component
                dx = -gy_val / length
                dy = gx_val / length
        
        elif gradient_source == "Image Edges":
            # Use image gradients (RGB edge detection)
            kernel_size = 5
            y_min = max(0, gy - kernel_size)
            y_max = min(h, gy + kernel_size + 1)
            x_min = max(0, gx - kernel_size)
            x_max = min(w, gx + kernel_size + 1)
            
            # Use the mask gradients as proxy for edge direction
            local_mask = mask[y_min:y_max, x_min:x_max].astype(np.float32)
            
            if local_mask.size > 0:
                local_gx = cv2.Sobel(local_mask, cv2.CV_64F, 1, 0, ksize=3)
                local_gy = cv2.Sobel(local_mask, cv2.CV_64F, 0, 1, ksize=3)
                
                center_y = min(kernel_size, gy - y_min)
                center_x = min(kernel_size, gx - x_min)
                
                if center_y < local_gy.shape[0] and center_x < local_gx.shape[1]:
                    gx_val = local_gx[center_y, center_x]
                    gy_val = local_gy[center_y, center_x]
                else:
                    gx_val = np.mean(local_gx)
                    gy_val = np.mean(local_gy)
            else:
                gx_val, gy_val = 1.0, 0.0
            
            length = np.sqrt(gx_val**2 + gy_val**2)
            if length < 1e-6:
                dx, dy = 1.0, 0.0
            else:
                dx = -gy_val / length
                dy = gx_val / length
        
        elif gradient_source == "Radial from Center":
            # Direction points radially from image center through grasp point
            center_x, center_y = w // 2, h // 2
            dx = gx - center_x
            dy = gy - center_y
            length = np.sqrt(dx**2 + dy**2)
            
            if length < 1e-6:
                dx, dy = 1.0, 0.0  # Fallback: horizontal
            else:
                dx /= length
                dy /= length
        
        else:
            # Default fallback
            dx, dy = 1.0, 0.0
        
        # Cast both directions from grasp point
        end1_x, end1_y = self._cast_single_ray(mask, gx, gy, dx, dy, max_dist)
        end2_x, end2_y = self._cast_single_ray(mask, gx, gy, -dx, -dy, max_dist)
        
        # Calculate total line length
        total_length = np.sqrt((end1_x - end2_x)**2 + (end1_y - end2_y)**2)
        
        return end1_x, end1_y, end2_x, end2_y, total_length
        """
        Cast a ray FROM grasp_point in perpendicular direction using various gradient methods.
        Does NOT necessarily go through CoG.
        
        Args:
            mask: Binary mask where 255=object, 0=background
            grasp_point: (x, y) tuple on the contour
            grad_x: X-component of gradient map
            grad_y: Y-component of gradient map
            gradient_source: Method to determine direction ("Depth Gradients", "Image Edges", "Radial from Center")
            max_dist: Maximum distance to search
            
        Returns:
            (end1_x, end1_y, end2_x, end2_y, total_length)
        """
        h, w = mask.shape
        gx, gy = grasp_point
        
        if gradient_source == "Depth Gradients":
            # Use depth gradient at this point (most reliable for depth-based grasping)
            if 0 <= gy < grad_y.shape[0] and 0 <= gx < grad_x.shape[1]:
                gx_val = grad_x[gy, gx]
                gy_val = grad_y[gy, gx]
            else:
                # Fallback to sampling around the point
                kernel_size = 3
                y_min = max(0, gy - kernel_size)
                y_max = min(h, gy + kernel_size + 1)
                x_min = max(0, gx - kernel_size)
                x_max = min(w, gx + kernel_size + 1)
                
                gx_val = np.mean(grad_x[y_min:y_max, x_min:x_max])
                gy_val = np.mean(grad_y[y_min:y_max, x_min:x_max])
            
            # Calculate perpendicular direction
            length = np.sqrt(gx_val**2 + gy_val**2)
            if length < 1e-6:
                # Fallback to radial
                dx = gx - w // 2
                dy = gy - h // 2
                length = np.sqrt(dx**2 + dy**2)
                if length < 1e-6:
                    dx, dy = 1.0, 0.0
                else:
                    dx /= length
                    dy /= length
            else:
                # Perpendicular to gradient: swap and negate one component
                dx = -gy_val / length
                dy = gx_val / length
        
        elif gradient_source == "Image Edges":
            # Use image gradients (RGB edge detection)
            kernel_size = 5
            y_min = max(0, gy - kernel_size)
            y_max = min(h, gy + kernel_size + 1)
            x_min = max(0, gx - kernel_size)
            x_max = min(w, gx + kernel_size + 1)
            
            # Use the mask gradients as proxy for edge direction
            local_mask = mask[y_min:y_max, x_min:x_max].astype(np.float32)
            
            if local_mask.size > 0:
                local_gx = cv2.Sobel(local_mask, cv2.CV_64F, 1, 0, ksize=3)
                local_gy = cv2.Sobel(local_mask, cv2.CV_64F, 0, 1, ksize=3)
                
                center_y = min(kernel_size, gy - y_min)
                center_x = min(kernel_size, gx - x_min)
                
                if center_y < local_gy.shape[0] and center_x < local_gx.shape[1]:
                    gx_val = local_gx[center_y, center_x]
                    gy_val = local_gy[center_y, center_x]
                else:
                    gx_val = np.mean(local_gx)
                    gy_val = np.mean(local_gy)
            else:
                gx_val, gy_val = 1.0, 0.0
            
            length = np.sqrt(gx_val**2 + gy_val**2)
            if length < 1e-6:
                dx, dy = 1.0, 0.0
            else:
                dx = -gy_val / length
                dy = gx_val / length
        
        elif gradient_source == "Radial from Center":
            # Direction points radially from image center through grasp point
            center_x, center_y = w // 2, h // 2
            dx = gx - center_x
            dy = gy - center_y
            length = np.sqrt(dx**2 + dy**2)
            
            if length < 1e-6:
                dx, dy = 1.0, 0.0  # Fallback: horizontal
            else:
                dx /= length
                dy /= length
        
        else:
            # Default fallback
            dx, dy = 1.0, 0.0
        
        # Cast both directions from grasp point
        end1_x, end1_y = self._cast_single_ray(mask, gx, gy, dx, dy, max_dist)
        end2_x, end2_y = self._cast_single_ray(mask, gx, gy, -dx, -dy, max_dist)
        
        # Calculate total line length
        total_length = np.sqrt((end1_x - end2_x)**2 + (end1_y - end2_y)**2)
        
        return end1_x, end1_y, end2_x, end2_y, total_length
    
    def _cast_ray_through_cog(self, mask, grasp_point, cog, max_dist=500):
        """
        Cast a ray FROM CoG through grasp_point, finding both edges.
        The line is defined by CoG->grasp_point direction.
        
        Args:
            mask: Binary mask where 255=object, 0=background
            grasp_point: (x, y) tuple defining the line direction
            cog: (x, y) tuple of the Center of Gravity (starting point)
            max_dist: Maximum distance to search
            
        Returns:
            (end1_x, end1_y, end2_x, end2_y, total_length)
        """
        h, w = mask.shape
        gx, gy = grasp_point
        cx, cy = cog
        
        # Calculate direction vector FROM CoG TO grasp point
        dx = gx - cx
        dy = gy - cy
        length = np.sqrt(dx**2 + dy**2)
        
        # Normalize direction (handle case where grasp point == CoG)
        if length < 1e-6:
            # If grasp point is at CoG, use perpendicular to find a line
            # This is a degenerate case, return zero-length grasp
            return cx, cy, cx, cy, 0.0
        
        dx /= length
        dy /= length
        
        # Cast ray FROM CoG in direction of grasp point (forward)
        end1_x, end1_y = self._cast_single_ray(mask, cx, cy, dx, dy, max_dist)
        
        # Cast ray FROM CoG in opposite direction (backward)
        end2_x, end2_y = self._cast_single_ray(mask, cx, cy, -dx, -dy, max_dist)
        
        # Calculate total line length
        total_length = np.sqrt((end1_x - end2_x)**2 + (end1_y - end2_y)**2)
        
        return end1_x, end1_y, end2_x, end2_y, total_length
    
    def _cast_single_ray(self, mask, start_x, start_y, dx, dy, max_dist):
        """
        Cast a single ray in direction (dx, dy) until hitting edge or boundary.
        
        Returns:
            (end_x, end_y) coordinates where ray stopped
        """
        h, w = mask.shape
        curr_x, curr_y = float(start_x), float(start_y)
        
        for _ in range(max_dist):
            curr_x += dx
            curr_y += dy
            ix, iy = int(round(curr_x)), int(round(curr_y))
            
            # Stop at image boundaries
            if ix < 0 or ix >= w or iy < 0 or iy >= h:
                return ix, iy
            
            # Stop when hitting background (edge of object)
            if mask[iy, ix] == 0:
                return ix, iy
        
        return int(curr_x), int(curr_y)

    def get_combined_saliency_mask_hybrid(self, image, depth_map, depth_percentile=60):
        """
        HYBRID APPROACH: Uses depth to constrain search region, then edges to find object.
        """
        # 1. DEPTH-BASED ROI
        depth_threshold = np.percentile(depth_map, depth_percentile)
        depth_roi = (depth_map >= depth_threshold).astype(np.uint8) * 255
        
        kernel_depth = np.ones((7, 7), np.uint8)
        depth_roi = cv2.morphologyEx(depth_roi, cv2.MORPH_CLOSE, kernel_depth)
        depth_roi = cv2.morphologyEx(depth_roi, cv2.MORPH_OPEN, kernel_depth)

        # 2. EDGE DETECTION (RGB input -> Gray)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)

        # 3. EDGES WITHIN ROI
        edges_in_roi = cv2.bitwise_and(edges, edges, mask=depth_roi)
        kernel_dilate = np.ones((5, 5), np.uint8)
        edges_dilated = cv2.dilate(edges_in_roi, kernel_dilate, iterations=3)

        # 4. COMBINE
        combined = cv2.bitwise_and(depth_roi, edges_dilated)
        depth_contribution = depth_roi.astype(np.float32) * 0.2
        edge_contribution = edges_dilated.astype(np.float32) * 0.8
        combined_blend = (depth_contribution + edge_contribution).astype(np.uint8)
        _, binary_mask = cv2.threshold(combined_blend, 100, 255, cv2.THRESH_BINARY)

        # 5. CLEANUP
        kernel_close = np.ones((7, 7), np.uint8)
        kernel_open = np.ones((5, 5), np.uint8)
        object_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel_open)

        # 6. CONTOUR
        contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        object_contour = None
        
        if contours:
            total_area = object_mask.shape[0] * object_mask.shape[1]
            valid_contours = [c for c in contours if 0.005 * total_area < cv2.contourArea(c) < 0.40 * total_area]
            
            if valid_contours:
                object_contour = max(valid_contours, key=cv2.contourArea)
            else:
                object_contour = max(contours, key=cv2.contourArea)
            
            object_mask = np.zeros_like(object_mask)
            cv2.drawContours(object_mask, [object_contour], -1, 255, thickness=cv2.FILLED)

        # 7. GRADIENTS
        depth_uint8 = (depth_map * 255).astype(np.uint8)
        grad_x = cv2.Sobel(depth_uint8, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(depth_uint8, cv2.CV_64F, 0, 1, ksize=5)
        
        return object_mask, object_contour, grad_x, grad_y, edges_in_roi

    def detect(self, rgb: np.ndarray, depth: np.ndarray, num_grasps: int = 10, 
               depth_percentile: int = 60, candidate_multiplier: int = 3,
               min_length: float = 100, max_length: float = 1000,
               ray_algorithm: str = "Through CoG", cog_boost: float = 0.0,
               gradient_source: str = "Depth Gradients"):
        h, w = rgb.shape[:2]
        debug = {}
        
        mask, contour, grad_x, grad_y, edges_roi = self.get_combined_saliency_mask_hybrid(
            rgb, depth, depth_percentile
        )
        
        depth_grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        depth_grad_norm = (depth_grad_mag - depth_grad_mag.min()) / (depth_grad_mag.max() - depth_grad_mag.min() + 1e-8)
        
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges_raw = cv2.Canny(gray, 50, 150)
        edges_normalized = edges_raw.astype(np.float32) / 255.0

        # Calculate Center of Gravity
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cog_x, cog_y = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        else:
            cog_x, cog_y = w // 2, h // 2
        
        # DEBUG: Check if mask is valid
        mask_area = np.count_nonzero(mask)
        debug['mask_area'] = mask_area
        debug['mask_valid'] = mask_area > 0
            
        debug['edges'] = edges_roi
        debug['depth_grad'] = depth_grad_norm
        debug['mask'] = mask
        debug['cog'] = (cog_x, cog_y)
        debug['mask_area'] = np.count_nonzero(mask)
        
        if contour is not None:
            contour_points = contour.reshape(-1, 2)
            indices = np.linspace(0, len(contour_points) - 1, min(len(contour_points), 100), dtype=int)
            candidate_points = contour_points[indices]
            debug['num_contour_points'] = len(candidate_points)
        else:
            candidate_points = [[w//2, h//2]]
            debug['num_contour_points'] = 0
        
        debug['num_contour_points'] = len(candidate_points)

        # STEP 1: Calculate quality scores for ALL candidates
        preliminary_candidates = []
        max_dist_scene = np.sqrt(w**2 + h**2)

        for point in candidate_points:
            px, py = point
            if not (0 <= py < h and 0 <= px < w): 
                continue
            
            # Calculate quality scores
            e_q = edges_normalized[py, px]
            d_q = depth_grad_norm[py, px]
            dist_cog = np.sqrt((px - cog_x)**2 + (py - cog_y)**2)
            c_q = 1 - (dist_cog / max_dist_scene)
            
            combined = (self.edge_weight * e_q + self.depth_weight * d_q + self.cog_weight * c_q)
            
            preliminary_candidates.append({
                'point': (px, py),
                'combined_quality': combined,
                'edge_quality': e_q,
                'depth_quality': d_q,
                'cog_quality': c_q
            })
        
        debug['num_preliminary'] = len(preliminary_candidates)
        
        # STEP 2: Sort by quality and take top candidates
        preliminary_candidates.sort(key=lambda x: x['combined_quality'], reverse=True)
        top_candidates = preliminary_candidates[:num_grasps * candidate_multiplier]  # Get candidates for ray casting
        
        debug['num_evaluated'] = len(top_candidates)
        
        # STEP 3: Do ray casting ONLY on top quality candidates
        grasp_candidates = []
        invalid_grasps = {'too_short': 0, 'too_long': 0, 'zero_length': 0}
        
        for candidate in top_candidates:
            px, py = candidate['point']
            
            # Choose ray casting algorithm
            if ray_algorithm == "Through CoG":
                end1_x, end1_y, end2_x, end2_y, line_length = self._cast_ray_through_cog(
                    mask, (px, py), (cog_x, cog_y)
                )
            else:  # "Direct Line with CoG Boost"
                end1_x, end1_y, end2_x, end2_y, line_length = self._cast_ray_direct_line(
                    mask, (px, py), grad_x, grad_y, contour, gradient_source
                )
            
            # DEBUG: Track why grasps are rejected
            if line_length <= min_length:
                if line_length == 0:
                    invalid_grasps['zero_length'] += 1
                else:
                    invalid_grasps['too_short'] += 1
                continue
            elif line_length >= max_length:
                invalid_grasps['too_long'] += 1
                continue
            
            # Calculate angle from the line direction
            dx_line = end1_x - end2_x
            dy_line = end1_y - end2_y
            angle = np.arctan2(dy_line, dx_line)
            
            # Center point is the midpoint of the line
            cx, cy = (end1_x + end2_x) / 2, (end1_y + end2_y) / 2
            
            # Calculate CoG proximity boost (for Direct Line algorithm)
            if ray_algorithm == "Direct Line with CoG Boost":
                dist_to_cog = np.sqrt((cx - cog_x)**2 + (cy - cog_y)**2)
                max_dist_scene = np.sqrt(w**2 + h**2)
                proximity_score = 1 - (dist_to_cog / max_dist_scene)
                boosted_score = line_length - (cog_boost * proximity_score * 500)  # Subtract boost (lower is better)
            else:
                boosted_score = line_length
            
            grasp_candidates.append(GraspCandidate(
                x=int(cx), y=int(cy), 
                angle=angle, 
                width=float(line_length),
                edge_quality=float(candidate['edge_quality']), 
                depth_quality=float(candidate['depth_quality']),
                cog_quality=float(candidate['cog_quality']), 
                combined_quality=float(candidate['combined_quality']),
                line_length=float(boosted_score)  # Store boosted score for ranking
            ))
        
        # STEP 4: Sort by line length (SHORTEST lines are best grasps)
        grasp_candidates.sort(key=lambda x: x.line_length)
        
        # Add debug info including top candidates visualization
        debug['num_valid'] = len(grasp_candidates)
        debug['invalid_grasps'] = invalid_grasps
        debug['has_contour'] = contour is not None
        debug['top_candidates'] = top_candidates[:20] if len(top_candidates) >= 20 else top_candidates  # Store for visualization
        
        return grasp_candidates[:num_grasps], debug

# ==========================================
# 2. HELPER FUNCTIONS & MODEL LOADING
# ==========================================

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def manage_model_memory(new_model_name):
    """
    Aggressive memory cleanup to prevent crashes when switching models
    on limited VRAM (like 8GB Laptop GPU).
    """
    if 'model_name' in st.session_state and st.session_state.model_name != new_model_name:
        st.toast(f"Switching from {st.session_state.model_name} to {new_model_name}...", icon="🧹")
        
        if 'depth_wrapper' in st.session_state:
            # Delete the object
            del st.session_state.depth_wrapper
            
            # Force Python garbage collection
            gc.collect()
            
            # Clear CUDA cache (Essential for MiDaS switch)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            st.session_state.model_loaded = False
    
    st.session_state.model_name = new_model_name

class DepthModelWrapper:
    def __init__(self, model_label, device):
        self.device = device
        self.model_label = model_label
        self.processor = None
        self.model = None
        self.transform = None
        self.model_family = None

        # --- MODEL DEFINITIONS ---
        # Map user friendly names to HF IDs or Hub Names
        MODEL_MAP = {
            "DepthAnythingV2-Small": {"type": "da", "id": "depth-anything/Depth-Anything-V2-Small-hf"},
            "DepthAnythingV2-Base":  {"type": "da", "id": "depth-anything/Depth-Anything-V2-Base-hf"},
            "DepthAnythingV2-Large": {"type": "da", "id": "depth-anything/Depth-Anything-V2-Large-hf"},
            "MiDaS-Small (v2.1)":    {"type": "midas", "id": "MiDaS_small"},
            "MiDaS-Hybrid (DPT)":    {"type": "midas", "id": "DPT_Hybrid"},
            "MiDaS-Large (DPT)":     {"type": "midas", "id": "DPT_Large"},
        }

        config = MODEL_MAP.get(model_label)
        if not config:
            raise ValueError(f"Unknown model: {model_label}")

        self.model_family = config["type"]

        try:
            if self.model_family == "da":
                # HuggingFace Loading
                self.processor = AutoImageProcessor.from_pretrained(config["id"])
                self.model = AutoModelForDepthEstimation.from_pretrained(config["id"]).to(device)
            
            elif self.model_family == "midas":
                # Torch Hub Loading
                self.model = torch.hub.load("intel-isl/MiDaS", config["id"]).to(device)
                
                # CRITICAL: Load correct transforms based on model type
                # Small uses 'small_transform', DPT uses 'dpt_transform'
                midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
                
                if config["id"] == "MiDaS_small":
                    self.transform = midas_transforms.small_transform
                else:
                    self.transform = midas_transforms.dpt_transform

            self.model.eval()
            
        except Exception as e:
            st.error(f"Failed to load model {model_label}. Error: {e}")
            raise e

    def predict(self, image_np):
        h, w = image_np.shape[:2]
        
        with torch.no_grad():
            if self.model_family == "da":
                pil_img = Image.fromarray(image_np)
                inputs = self.processor(images=pil_img, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                depth = self.model(**inputs).predicted_depth
                depth = F.interpolate(depth.unsqueeze(1), size=(h, w), mode='bicubic', align_corners=False).squeeze()
            
            elif self.model_family == "midas":
                input_batch = self.transform(image_np).to(self.device)
                prediction = self.model(input_batch)
                depth = F.interpolate(prediction.unsqueeze(1), size=(h, w), mode='bicubic', align_corners=False).squeeze()
        
        depth = depth.cpu().numpy()
        # Normalize 0-1
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth

def create_interactive_plot(image_rgb, grasps, cog=None, show_rays=True):
    fig = go.Figure()

    # Add Image
    fig.add_trace(go.Image(z=image_rgb))

    # Add CoG
    if cog:
        fig.add_trace(go.Scatter(
            x=[cog[0]], y=[cog[1]], mode='markers',
            marker=dict(color='cyan', symbol='x', size=15, line=dict(width=2, color='white')),
            name='Center of Gravity',
            hoverinfo='text', hovertext='Center of Gravity'
        ))

    # Add Grasps
    for i, g in enumerate(grasps):
        corners = g.get_corners()
        corners_plot = np.vstack([corners, corners[0]])
        
        # Best grasp is lime, others are red
        color = 'lime' if i == 0 else 'red'
        
        info = (f"Rank: #{i+1}<br>"
                f"Line Length: {g.line_length:.1f}px<br>"
                f"Combined Quality: {g.combined_quality:.2f}<br>"
                f"Edge: {g.edge_quality:.2f}<br>"
                f"Depth: {g.depth_quality:.2f}<br>"
                f"CoG: {g.cog_quality:.2f}<br>"
                f"Width: {g.width:.1f}px")

        # Draw grasp rectangle
        fig.add_trace(go.Scatter(
            x=corners_plot[:, 0], y=corners_plot[:, 1],
            mode='lines',
            line=dict(color=color, width=3 if i==0 else 2),
            name=f'Grasp #{i+1}',
            text=info,
            hoverinfo='text',
            showlegend=False
        ))
        
        # Draw ray line from CoG through grasp (for top 3 grasps)
        if show_rays and i < 3 and cog:
            # Calculate the line endpoints based on grasp angle and width
            cos_a = np.cos(g.angle)
            sin_a = np.sin(g.angle)
            half_w = g.width / 2
            
            # Line endpoints
            x1 = g.x - half_w * cos_a
            y1 = g.y - half_w * sin_a
            x2 = g.x + half_w * cos_a
            y2 = g.y + half_w * sin_a
            
            ray_color = 'yellow' if i == 0 else 'orange'
            
            fig.add_trace(go.Scatter(
                x=[x1, cog[0], x2], 
                y=[y1, cog[1], y2],
                mode='lines',
                line=dict(color=ray_color, width=2, dash='dot'),
                name=f'Ray #{i+1}',
                hoverinfo='skip',
                showlegend=False
            ))

    fig.update_layout(
        width=700, height=500,
        margin=dict(l=0, r=0, b=0, t=0),
        xaxis=dict(visible=False, range=[0, image_rgb.shape[1]]),
        yaxis=dict(visible=False, range=[image_rgb.shape[0], 0], scaleanchor="x"),
    )
    return fig

# ==========================================
# 3. STREAMLIT APP LAYOUT
# ==========================================

st.set_page_config(page_title="Grasp Detection Lab", layout="wide")

st.title("Grasp Detection Sandbox")
st.markdown("Interactive pipeline for tuning weights and comparing depth models for heuristic grasping")
# st.info("**TWO-STAGE ALGORITHM**: (1) Quality scores select top candidates → (2) Ray casting finds shortest line through CoG")

# --- SIDEBAR: Configuration ---
with st.sidebar:
    st.header("Configuration")
    
    # Expanded Model Options
    model_option = st.selectbox(
        "Depth Model", 
        [
            "DepthAnythingV2-Small", 
            "DepthAnythingV2-Base",
            "DepthAnythingV2-Large", 
            "MiDaS-Small (v2.1)",
            "MiDaS-Hybrid (DPT)",
            "MiDaS-Large (DPT)"
        ],
        help="Switching models will unload the previous one to save VRAM. Large models may be slow."
    )
    
    st.divider()
    
    # --- WEIGHTS ---
    st.subheader("Heuristic Weights")
    st.caption("Used to SELECT TOP candidates (Stage 1)")
    w_edge = st.slider("Edge Weight", 0.0, 1.0, 0.20, 0.05)
    w_depth = st.slider("Depth Gradient Weight", 0.0, 1.0, 0.40, 0.05)
    w_cog = st.slider("CoG Weight", 0.0, 1.0, 0.40, 0.05)
    
    total = w_edge + w_depth + w_cog
    if total > 0:
        st.caption(f"Normalized: Edge {w_edge/total:.2f} | Depth {w_depth/total:.2f} | CoG {w_cog/total:.2f}")
    
    st.divider()

    # --- HYBRID MASK SETTINGS ---
    st.subheader("Hybrid Mask Settings")
    st.caption("Controls the 'Saliency Mask' generation.")
    
    depth_percentile = st.slider(
        "Depth Percentile", 0, 100, 60, step=5,
        help="Higher values select only the closest objects. (60 = top 40% closest)"
    )

    st.divider()
    
    # --- RAY CASTING ALGORITHM ---
    st.subheader("Ray Casting Algorithm")
    ray_algorithm = st.selectbox(
        "Algorithm Type",
        ["Through CoG", "Direct Line with CoG Boost"],
        help="Through CoG: Line must pass through center. Direct Line: Straight across with proximity bonus."
    )
    
    if ray_algorithm == "Direct Line with CoG Boost":
        cog_boost_weight = st.slider(
            "CoG Proximity Boost", 0.0, 10.0, 2.0, 0.5,
            help="Higher values favor grasps closer to CoG. Multiplier applied to proximity score."
        )
        
        gradient_source = st.selectbox(
            "Gradient Source",
            ["Contour Direction (80px avg)", "Depth Gradients", "Image Edges", "Radial from Center"],
            help="Method for determining perpendicular direction. Contour uses 80px segment average."
        )
    else:
        cog_boost_weight = 0.0
        gradient_source = "Contour Direction (80px avg)"
    
    st.divider()

    # --- GRASP FILTERING ---
    st.subheader("Grasp Length Filtering")
    col1, col2 = st.columns(2)
    with col1:
        min_grasp_length = st.number_input("Min Length (px)", 1, 10000, 100, 10, 
                                           help="Grasps shorter than this are rejected")
    with col2:
        max_grasp_length = st.number_input("Max Length (px)", 100, 10000, 1000, 50,
                                           help="Grasps longer than this are rejected")
    
    st.divider()
    
    num_grasps = st.number_input("Max Grasps to Return", 1, 50, 5)
    candidate_multiplier = st.slider(
        "Candidate Multiplier", 1, 10, 3, 
        help="Evaluate (num_grasps × multiplier) top candidates before selecting shortest lines"
    )
    
    run_btn = st.button("Run Pipeline", type="primary")

# --- INITIALIZATION ---
if 'history' not in st.session_state:
    st.session_state.history = []

manage_model_memory(model_option)

if 'depth_wrapper' not in st.session_state:
    with st.spinner(f"Loading {model_option} to GPU..."):
        try:
            st.session_state.depth_wrapper = DepthModelWrapper(model_option, get_device())
            st.session_state.model_loaded = True
        except Exception as e:
            st.stop() # Stop execution if model fails to load

# --- MAIN INTERFACE ---
st.subheader("Upload Image")
uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is None:
    st.info("Upload an image and click 'Run Pipeline' in the sidebar.")

if uploaded_file and run_btn and st.session_state.get('model_loaded'):
    # 1. Pre-processing
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 2. Pipeline Execution
    start_time = time.time()
    
    # Depth
    t0 = time.time()
    depth_map = st.session_state.depth_wrapper.predict(image_rgb)
    t_depth = (time.time() - t0) * 1000
    
    # Detection
    detector = ImprovedGraspDetector(w_edge, w_depth, w_cog)
    t1 = time.time()
    
    # Pass hybrid mask parameters
    grasps, debug_info = detector.detect(
        image_rgb, depth_map, num_grasps,
        depth_percentile=depth_percentile,
        candidate_multiplier=candidate_multiplier,
        min_length=min_grasp_length,
        max_length=max_grasp_length,
        ray_algorithm=ray_algorithm,
        cog_boost=cog_boost_weight,
        gradient_source=gradient_source
    )
    t_detect = (time.time() - t1) * 1000
    
    total_time = (time.time() - start_time) * 1000
    
    # 3. Save to History
    result_entry = {
        "id": len(st.session_state.history) + 1,
        "model": model_option,
        "weights": (w_edge, w_depth, w_cog),
        "params": f"Percentile: {depth_percentile}, Multiplier: {candidate_multiplier}, Algorithm: {ray_algorithm}",
        "grasps": grasps,
        "image": image_rgb,
        "depth": depth_map,
        "mask": debug_info['mask'],
        "edges": debug_info['edges'], 
        "grads": debug_info['depth_grad'],
        "time": total_time,
        "cog": debug_info.get('cog'),
        "num_evaluated": debug_info.get('num_evaluated', 0),
        "num_valid": debug_info.get('num_valid', 0),
        "min_length": min_grasp_length,
        "max_length": max_grasp_length,
        "invalid_grasps": debug_info.get('invalid_grasps', {}),
        "has_contour": debug_info.get('has_contour', False),
        "top_candidates": debug_info.get('top_candidates', []),
        "ray_algorithm": ray_algorithm
    }
    st.session_state.history.insert(0, result_entry)
    
    # Show warnings if no grasps found
    if len(grasps) == 0:
        st.error("No valid grasps found!")
        
        with st.expander("Debug Information", expanded=True):
            st.write("**Mask Statistics:**")
            st.write(f"- Mask area (non-zero pixels): {debug_info.get('mask_area', 0)}")
            st.write(f"- Has valid contour: {debug_info.get('has_contour', False)}")
            st.write(f"- Contour points sampled: {debug_info.get('num_contour_points', 0)}")
            
            st.write("\n**Pipeline Statistics:**")
            st.write(f"- Preliminary candidates: {debug_info.get('num_preliminary', 0)}")
            st.write(f"- Top candidates evaluated: {debug_info.get('num_evaluated', 0)}")
            st.write(f"- Valid grasps found: {debug_info.get('num_valid', 0)}")
            
            invalid = debug_info.get('invalid_grasps', {})
            if invalid:
                st.write("\n**Rejection Reasons:**")
                st.write(f"- Too short (≤{min_grasp_length}px): {invalid.get('too_short', 0)}")
                st.write(f"- Too long (≥{max_grasp_length}px): {invalid.get('too_long', 0)}")
                st.write(f"- Zero length: {invalid.get('zero_length', 0)}")
            
            st.write("\n**Troubleshooting Tips:**")
            if debug_info.get('mask_area', 0) < 100:
                st.warning("- Mask is too small! Try LOWERING the depth percentile (e.g., 40-50)")
            if not debug_info.get('has_contour', False):
                st.warning("- No contour found! The mask might be empty or too fragmented")
            if invalid.get('zero_length', 0) > 0:
                st.warning("- Zero-length lines detected! Grasp points might be at CoG. Try adjusting weights or depth percentile.")
            if invalid.get('too_short', 0) > invalid.get('too_long', 0) + invalid.get('zero_length', 0):
                st.warning(f"- Most grasps too short! Try LOWERING 'Min Length' below {min_grasp_length}px")
            if invalid.get('too_long', 0) > 10:
                st.warning(f"- Many grasps too long! Try INCREASING 'Max Length' above {max_grasp_length}px")

# --- DISPLAY CURRENT RESULTS ---
if st.session_state.history:
    current = st.session_state.history[0]
    
    st.subheader(f"Results (Run #{current['id']})")
    
    # Show warning if no grasps
    if len(current['grasps']) == 0:
        st.warning("No valid grasps found for this image. See debug information below.")
    
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Time", f"{current['time']:.0f} ms")
    m2.metric("Candidates Evaluated", current.get('num_evaluated', 'N/A'))
    m3.metric("Valid Grasps", current.get('num_valid', len(current['grasps'])))
    m4.metric("Returned", len(current['grasps']))
    m5.metric("Detection Time", f"{t_detect:.0f} ms")
    
    # Show rejection statistics if any grasps were filtered
    invalid = current.get('invalid_grasps', {})
    total_rejected = sum(invalid.values())
    
    if total_rejected > 0:
        with st.expander(f"{total_rejected} candidates were filtered out - Click to see why", expanded=False):
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("Too Short", invalid.get('too_short', 0), 
                      help=f"Lines ≤ {current.get('min_length', 10)}px")
            rc2.metric("Too Long", invalid.get('too_long', 0),
                      help=f"Lines ≥ {current.get('max_length', 400)}px")
            rc3.metric("Zero Length", invalid.get('zero_length', 0),
                      help="Grasp point coincides with CoG")
            rc4.metric("Total Rejected", total_rejected)
            
            st.markdown("**Quick Fixes:**")
            if invalid.get('too_short', 0) > 5:
                st.info(f"Many too short → Lower 'Min Length' in sidebar (currently {current.get('min_length', 10)}px)")
            if invalid.get('too_long', 0) > 5:
                st.info(f"Many too long → Increase 'Max Length' in sidebar (currently {current.get('max_length', 400)}px)")
            if invalid.get('zero_length', 0) > 2:
                st.info("Zero-length detected → Adjust depth percentile or quality weights")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Original Image**")
        st.image(current['image'], use_container_width=True)
    with c2:
        if len(current['grasps']) > 0:
            st.markdown("**Interactive Result (Hover for details)**")
            fig = create_interactive_plot(current['image'], current['grasps'], current['cog'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("**No Grasps Found - Showing Mask**")
            # Show the mask to help debug
            mask_colored = cv2.cvtColor(current['mask'], cv2.COLOR_GRAY2RGB)
            mask_colored[current['mask'] > 0] = [100, 255, 100]  # Green for object
            overlay = cv2.addWeighted(current['image'], 0.6, mask_colored, 0.4, 0)
            
            # Draw CoG
            cog = current['cog']
            cv2.circle(overlay, cog, 10, (0, 255, 255), -1)
            cv2.circle(overlay, cog, 10, (0, 0, 0), 2)
            
            st.image(overlay, use_container_width=True)
            st.caption("Green = detected object, Cyan dot = Center of Gravity")

    # --- GRASP RANKING TABLE ---
    if len(current['grasps']) > 0:
        st.markdown("### Grasp Rankings (by Line Length)")
        ranking_data = []
        for i, g in enumerate(current['grasps']):
            ranking_data.append({
                "Rank": i + 1,
                "Line Length (px)": f"{g.line_length:.1f}",
                "Quality Score": f"{g.combined_quality:.3f}",
                "Width (px)": f"{g.width:.1f}",
                "Center": f"({g.x}, {g.y})"
            })
        st.table(ranking_data)

    # --- NEW INTERMEDIATE VISUALIZATIONS ---
    st.markdown("### Detection Pipeline Stages")
    
    stage_cols = st.columns(3)
    
    # Stage 1: Top Candidates
    with stage_cols[0]:
        st.markdown("**Stage 1: Top Candidates**")
        st.caption(f"Top {len(current.get('top_candidates', []))} by quality score")
        
        top_cand_img = current['image'].copy()
        cog = current['cog']
        
        # Draw CoG
        cv2.circle(top_cand_img, cog, 8, (0, 255, 255), -1)
        cv2.circle(top_cand_img, cog, 8, (0, 0, 0), 2)
        
        # Draw top candidates as circles
        for i, cand in enumerate(current.get('top_candidates', [])[:20]):
            px, py = cand['point']
            # Color gradient from green (best) to yellow (worst)
            color_intensity = int(255 * (1 - i/20))
            color = (color_intensity, 255, 0)  # Green to yellow
            cv2.circle(top_cand_img, (px, py), 5, color, -1)
            cv2.circle(top_cand_img, (px, py), 5, (0, 0, 0), 1)
        
        st.image(top_cand_img, use_container_width=True)
    
    # Stage 2: CoG Only
    with stage_cols[1]:
        st.markdown("**Stage 2: Center of Gravity**")
        st.caption("Reference point for ray casting")
        
        cog_only_img = current['image'].copy()
        cog = current['cog']
        
        # Draw large CoG marker
        cv2.circle(cog_only_img, cog, 15, (0, 255, 255), -1)
        cv2.circle(cog_only_img, cog, 15, (0, 0, 0), 3)
        
        # Draw crosshair
        cv2.line(cog_only_img, (cog[0]-25, cog[1]), (cog[0]+25, cog[1]), (0, 255, 255), 3)
        cv2.line(cog_only_img, (cog[0], cog[1]-25), (cog[0], cog[1]+25), (0, 255, 255), 3)
        
        st.image(cog_only_img, use_container_width=True)
    
    # Stage 3: Final Grasps
    with stage_cols[2]:
        st.markdown("**Stage 3: Final Grasps**")
        st.caption(f"Top {len(current['grasps'])} by {current.get('ray_algorithm', 'line length')}")
        
        if len(current['grasps']) > 0:
            final_img = current['image'].copy()
            cog = current['cog']
            
            # Draw CoG
            cv2.circle(final_img, cog, 8, (0, 255, 255), -1)
            
            # Draw grasps
            for i, g in enumerate(current['grasps']):
                corners = g.get_corners().astype(np.int32)
                color = (0, 255, 0) if i == 0 else (0, 0, 255)  # Green for best, red for others
                cv2.polylines(final_img, [corners], True, color, 3)
            
            st.image(final_img, use_container_width=True)
        else:
            st.image(current['image'], use_container_width=True)
            st.caption("No valid grasps found")
    
    # --- INTERMEDIATE STEPS ---
    st.markdown("### Mask Generation Pipeline")
    with st.expander("Show Pipeline Internals", expanded=False):
        ic1, ic2, ic3, ic4 = st.columns(4)
        
        ic1.markdown("**1. Depth Map**")
        ic1.image(current['depth'], clamp=True, use_container_width=True)
        
        ic2.markdown(f"**2. Mask (Percentile {depth_percentile})**")
        ic2.image(current['mask'], clamp=True, use_container_width=True)
        
        ic3.markdown("**3. Edges in ROI**")
        ic3.image(current['edges'], clamp=True, use_container_width=True)
        
        ic4.markdown("**4. Depth Gradients**")
        ic4.image(current['grads'], clamp=True, use_container_width=True)

    st.divider()

# --- HISTORY SECTION ---
st.header("Session History (Comparison)")

if len(st.session_state.history) > 1:
    for item in st.session_state.history[1:]:
        with st.container():
            st.markdown(f"**Run #{item['id']}** | Model: `{item['model']}` | Weights: {item['weights']} | {item['params']}")
            
            hc1, hc2, hc3 = st.columns([1, 1, 2])
            with hc1:
                st.image(item['image'], caption="Input")
            with hc2:
                st.image(item['mask'], caption="Generated Mask")
            with hc3:
                h_fig = create_interactive_plot(item['image'], item['grasps'], item['cog'])
                h_fig.update_layout(height=300, width=400)
                st.plotly_chart(h_fig, use_container_width=False)
            st.divider()
elif len(st.session_state.history) == 1:
    st.info("Run the pipeline again with different settings to see comparisons here.")