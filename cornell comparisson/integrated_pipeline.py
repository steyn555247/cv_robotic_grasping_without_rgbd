# ==========================================
# ENHANCED GRASP DETECTION WITH WHITE SURFACE CROPPING
# ==========================================
# Pipeline: Crop White Surface -> Mask -> Find COG -> Detect Grasps
# ==========================================

import cv2
import numpy as np

# ==========================================
# CONFIGURABLE PARAMETERS
# ==========================================

# Weights for quality scoring (must sum to ~1.0)
W_EDGE = 0.001              # Edge quality weight
W_DEPTH = 0.001             # Depth gradient weight
W_COG = 0.999               # CoG proximity weight

# Masking parameters
DEPTH_PERCENTILE = 30       # Depth percentile cutoff for background (0-100)

# White surface cropping parameters
ENABLE_WHITE_CROP = True    # Enable/disable white surface cropping
WHITE_LOWER = (200, 200, 200)  # Lower RGB threshold for white detection
WHITE_UPPER = (255, 255, 255)  # Upper RGB threshold for white detection
MIN_SURFACE_AREA = 0.1      # Minimum area ratio for valid surface

# Ray casting parameters
RAY_ALGORITHM = "Direct Line with CoG Boost"
COG_BOOST_VALUE = 3.75
GRADIENT_SOURCE = "Contour Direction (80px avg)"

# Filtering parameters
MIN_GRASP_LENGTH = 1
MAX_GRASP_LENGTH = 1000

# Output parameters
NUM_OUTPUT_GRASPS = 1
CANDIDATE_MULTIPLIER = 100

print("="*80)
print("ENHANCED PIPELINE WITH WHITE SURFACE CROPPING")
print("="*80)
print(f"White Crop Enabled: {ENABLE_WHITE_CROP}")
print(f"Weights: Edge={W_EDGE:.3f}, Depth={W_DEPTH:.3f}, CoG={W_COG:.3f}")
print(f"Depth Percentile: {DEPTH_PERCENTILE}")
print(f"Ray Algorithm: {RAY_ALGORITHM}")
print(f"CoG Boost: {COG_BOOST_VALUE}")
print(f"Gradient Source: {GRADIENT_SOURCE}")
print(f"Grasp Length Range: [{MIN_GRASP_LENGTH}, {MAX_GRASP_LENGTH}]")
print(f"Output Grasps: {NUM_OUTPUT_GRASPS}, Candidate Multiplier: {CANDIDATE_MULTIPLIER}")
print("="*80)


# ==========================================
# WHITE SURFACE CROPPING FUNCTION
# ==========================================

def crop_white_surface(image, lower_white=(200, 200, 200), upper_white=(255, 255, 255),
                       min_area_ratio=0.1):
    """
    Detect and crop the white surface from an image.

    Returns:
        cropped_image: Cropped RGB image
        crop_bbox: (x, y, w, h) bounding box
        crop_success: True if cropping succeeded
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Create mask for white regions
    rgb_mask = cv2.inRange(image, np.array(lower_white), np.array(upper_white))
    gray_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]
    combined_mask = cv2.bitwise_or(rgb_mask, gray_mask)

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find largest contour
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return image, (0, 0, image.shape[1], image.shape[0]), False

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Check minimum area
    image_area = image.shape[0] * image.shape[1]
    contour_area = cv2.contourArea(largest_contour)

    if contour_area < image_area * min_area_ratio:
        return image, (0, 0, image.shape[1], image.shape[0]), False

    # Add small padding
    padding_x = int(w * 0.02)
    padding_y = int(h * 0.02)
    x = max(0, x - padding_x)
    y = max(0, y - padding_y)
    w = min(image.shape[1] - x, w + 2 * padding_x)
    h = min(image.shape[0] - y, h + 2 * padding_y)

    cropped_image = image[y:y+h, x:x+w]
    return cropped_image, (x, y, w, h), True


# ==========================================
# ENHANCED CANDIDATE CLASS
# ==========================================

class Candidate:
    def __init__(self, x, y, angle, w, h=20.0, eq=0.0, dq=0.0, cq=0.0, comb=0.0, l=0.0):
        self.x = int(x)
        self.y = int(y)
        self.angle = angle
        self.width = float(w)
        self.height = h
        self.edge_quality = float(eq)
        self.depth_quality = float(dq)
        self.cog_quality = float(cq)
        self.combined_quality = float(comb)
        self.line_length = float(l)

    def get_corners(self):
        c = np.cos(self.angle)
        s = np.sin(self.angle)
        w, h = self.width, self.height

        pts = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]

        res = []
        for dx, dy in pts:
            px = self.x + dx * c - dy * s
            py = self.y + dx * s + dy * c
            res.append([px, py])

        return np.array(res)


# ==========================================
# ENHANCED GRASP DETECTOR
# ==========================================

class GraspDetector:
    def __init__(self, w_e, w_d, w_c):
        self.we = w_e
        self.wd = w_d
        self.wc = w_c

    def _ray_cast(self, mask, start, vec_x, vec_y, contour=None, mode="Depth Gradients", limit=500):
        h, w = mask.shape
        gx, gy = start
        dx, dy = 1.0, 0.0

        if mode == "Contour Direction (80px avg)":
            if contour is not None:
                pts = contour.reshape(-1, 2)
                dists = np.sqrt((pts[:, 0] - gx)**2 + (pts[:, 1] - gy)**2)
                idx = np.argmin(dists)

                n_pts = min(40, len(pts) // 4)
                half = n_pts // 2
                s_idx = (idx - half) % len(pts)
                e_idx = (idx + half) % len(pts)

                if s_idx < e_idx:
                    seg = pts[s_idx:e_idx+1]
                else:
                    seg = np.vstack([pts[s_idx:], pts[:e_idx+1]])

                if len(seg) > 2:
                    xs = seg[:, 0]
                    ys = seg[:, 1]
                    mx, my = np.mean(xs), np.mean(ys)
                    x_c = xs - mx
                    y_c = ys - my

                    Sxx = np.sum(x_c * x_c)
                    Sxy = np.sum(x_c * y_c)
                    Syy = np.sum(y_c * y_c)

                    if Sxx + Syy > 1e-6:
                        tr = Sxx + Syy
                        det = Sxx * Syy - Sxy * Sxy
                        l1 = tr/2 + np.sqrt(max(0, (tr/2)**2 - det))

                        if abs(Sxy) > 1e-6:
                            tx = l1 - Syy
                            ty = Sxy
                        elif abs(Sxx - l1) > 1e-6:
                            tx = Sxy
                            ty = l1 - Sxx
                        else:
                            tx = seg[-1, 0] - seg[0, 0]
                            ty = seg[-1, 1] - seg[0, 1]

                        l = np.sqrt(tx**2 + ty**2)
                        if l > 1e-6:
                            tx /= l; ty /= l
                            dx = -ty
                            dy = tx
                        else:
                            dx, dy = 1.0, 0.0
                    else:
                        dx, dy = 1.0, 0.0
                else:
                    if len(seg) >= 2:
                        tx = seg[-1, 0] - seg[0, 0]
                        ty = seg[-1, 1] - seg[0, 1]
                        l = np.sqrt(tx**2 + ty**2)
                        if l > 1e-6:
                            tx /= l; ty /= l
                            dx, dy = -ty, tx

        elif mode == "Depth Gradients":
            if 0 <= gy < vec_y.shape[0] and 0 <= gx < vec_x.shape[1]:
                val_x = vec_x[gy, gx]
                val_y = vec_y[gy, gx]
            else:
                ks = 3
                val_x = np.mean(vec_x[max(0, gy-ks):min(h, gy+ks+1), max(0, gx-ks):min(w, gx+ks+1)])
                val_y = np.mean(vec_y[max(0, gy-ks):min(h, gy+ks+1), max(0, gx-ks):min(w, gx+ks+1)])

            lenght = np.sqrt(val_x**2 + val_y**2)
            if lenght < 1e-6:
                dx = gx - w // 2
                dy = gy - h // 2
                lenght = np.sqrt(dx**2 + dy**2)
                if lenght > 1e-6: dx /= lenght; dy /= lenght
            else:
                dx = -val_y / lenght
                dy = val_x / lenght

        elif mode == "Image Edges":
            ks = 5
            ymin, ymax = max(0, gy - ks), min(h, gy + ks + 1)
            xmin, xmax = max(0, gx - ks), min(w, gx + ks + 1)

            local = mask[ymin:ymax, xmin:xmax].astype(np.float32)
            if local.size > 0:
                lx = cv2.Sobel(local, cv2.CV_64F, 1, 0, ksize=3)
                ly = cv2.Sobel(local, cv2.CV_64F, 0, 1, ksize=3)

                cy = min(ks, gy - ymin)
                cx = min(ks, gx - xmin)

                if cy < ly.shape[0] and cx < lx.shape[1]:
                    vx, vy = lx[cy, cx], ly[cy, cx]
                else:
                    vx, vy = np.mean(lx), np.mean(ly)

                l = np.sqrt(vx**2 + vy**2)
                if l > 1e-6:
                    dx = -vy / l
                    dy = vx / l

        elif mode == "Radial from Center":
            cx, cy = w//2, h//2
            dx = gx - cx
            dy = gy - cy
            l = np.sqrt(dx**2 + dy**2)
            if l > 1e-6:
                dx /= l
                dy /= l

        x1, y1 = self._trace_line(mask, gx, gy, dx, dy, limit)
        x2, y2 = self._trace_line(mask, gx, gy, -dx, -dy, limit)

        full_dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        return x1, y1, x2, y2, full_dist

    def _cog_ray(self, mask, gp, cog, limit=500):
        gx, gy = gp
        cx, cy = cog

        vx = gx - cx
        vy = gy - cy

        dist = np.sqrt(vx**2 + vy**2)
        if dist < 1e-6: return cx, cy, cx, cy, 0.0

        vx /= dist
        vy /= dist

        e1x, e1y = self._trace_line(mask, cx, cy, vx, vy, limit)
        e2x, e2y = self._trace_line(mask, cx, cy, -vx, -vy, limit)

        l = np.sqrt((e1x-e2x)**2 + (e1y-e2y)**2)
        return e1x, e1y, e2x, e2y, l

    def _trace_line(self, mask, sx, sy, dx, dy, limit):
        h, w = mask.shape
        cx, cy = float(sx), float(sy)

        GAP_MAX = 10
        gap = 0
        last_x, last_y = int(cx), int(cy)

        for _ in range(limit):
            cx += dx
            cy += dy
            ix, iy = int(round(cx)), int(round(cy))

            if ix < 0 or ix >= w or iy < 0 or iy >= h:
                return last_x, last_y

            if mask[iy, ix] == 0:
                gap += 1
                if gap > GAP_MAX:
                    return last_x, last_y
            else:
                gap = 0
                last_x, last_y = ix, iy

        return last_x, last_y

    def get_mask_data(self, img, dmap, pct=60):
        thresh = np.percentile(dmap, pct)
        droi = (dmap >= thresh).astype(np.uint8) * 255

        k1 = np.ones((7,7), np.uint8)
        droi = cv2.morphologyEx(droi, cv2.MORPH_CLOSE, k1)
        droi = cv2.morphologyEx(droi, cv2.MORPH_OPEN, k1)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 100)

        e_roi = cv2.bitwise_and(edges, edges, mask=droi)
        dilated = cv2.dilate(e_roi, np.ones((5,5), np.uint8), iterations=3)

        blend = (droi.astype(np.float32) * 0.2 + dilated.astype(np.float32) * 0.8).astype(np.uint8)
        _, bin_mask = cv2.threshold(blend, 100, 255, cv2.THRESH_BINARY)

        final_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

        conts, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main_c = None

        if conts:
            area_tot = final_mask.shape[0] * final_mask.shape[1]
            valid = [c for c in conts if 0.005 * area_tot < cv2.contourArea(c) < 0.40 * area_tot]
            if valid:
                main_c = max(valid, key=cv2.contourArea)
            else:
                main_c = max(conts, key=cv2.contourArea)

            final_mask = np.zeros_like(final_mask)
            cv2.drawContours(final_mask, [main_c], -1, 255, cv2.FILLED)

        d8 = (dmap * 255).astype(np.uint8)
        gx = cv2.Sobel(d8, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(d8, cv2.CV_64F, 0, 1, ksize=5)

        return final_mask, main_c, gx, gy, e_roi

    def process(self, rgb, depth, n_grasps=10, pct=60, mult=3, min_l=100, max_l=1000,
                algo="Through CoG", boost=0.0, grad_src="Depth Gradients",
                enable_crop=True, white_params=None):
        """
        Enhanced process with white surface cropping.

        Pipeline: Crop White Surface -> Mask -> Find COG -> Detect Grasps
        """

        info = {}
        crop_bbox = None

        # STEP 1: CROP WHITE SURFACE (if enabled)
        if enable_crop:
            if white_params is None:
                white_params = {
                    'lower': WHITE_LOWER,
                    'upper': WHITE_UPPER,
                    'min_area': MIN_SURFACE_AREA
                }

            cropped_rgb, crop_bbox, crop_success = crop_white_surface(
                rgb,
                lower_white=white_params['lower'],
                upper_white=white_params['upper'],
                min_area_ratio=white_params['min_area']
            )

            if crop_success:
                # Also crop the depth map
                x, y, w, h = crop_bbox
                cropped_depth = depth[y:y+h, x:x+w]

                # Use cropped images for processing
                rgb = cropped_rgb
                depth = cropped_depth

                info['cropped'] = True
                info['crop_bbox'] = crop_bbox
                info['original_size'] = (depth.shape[1] + w, depth.shape[0] + h)
                info['cropped_size'] = (w, h)
            else:
                info['cropped'] = False
        else:
            info['cropped'] = False

        # STEP 2: GET MASK AND COG (existing pipeline)
        h, w = rgb.shape[:2]
        mask, cont, gx, gy, ed = self.get_mask_data(rgb, depth, pct)

        mag = np.sqrt(gx**2 + gy**2)
        norm_g = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        enorm = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0

        # STEP 3: FIND COG
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
        else:
            cx, cy = w//2, h//2

        info['mask_area'] = np.count_nonzero(mask)
        info['mask_valid'] = info['mask_area'] > 0
        info['edges'] = ed
        info['depth_grad'] = norm_g
        info['mask'] = mask
        info['cog'] = (cx, cy)

        # STEP 4: SAMPLE GRASP CANDIDATE POINTS
        if cont is not None:
            pts = cont.reshape(-1, 2)
            idx = np.linspace(0, len(pts)-1, min(len(pts), 100), dtype=int)
            cands = pts[idx]
        else:
            cands = [[w//2, h//2]]

        info['num_contour_points'] = len(cands)

        # STEP 5: RANK CANDIDATES
        prelim = []
        diag = np.sqrt(w**2 + h**2)

        for p in cands:
            px, py = p
            if not (0 <= py < h and 0 <= px < w): continue

            eq = enorm[py, px]
            dq = norm_g[py, px]
            dist = np.sqrt((px - cx)**2 + (py - cy)**2)
            cq = 1 - (dist / diag)

            score = (self.we * eq + self.wd * dq + self.wc * cq)

            prelim.append({
                'p': (px, py),
                'score': score,
                'eq': eq, 'dq': dq, 'cq': cq
            })

        info['num_preliminary'] = len(prelim)
        prelim.sort(key=lambda x: x['score'], reverse=True)

        top = prelim[:n_grasps * mult]
        info['num_evaluated'] = len(top)

        # STEP 6: PERFORM GRASPING (ray casting)
        final_grasps = []
        rejects = {'too_short': 0, 'too_long': 0, 'zero_length': 0}

        for t in top:
            px, py = t['p']

            if algo == "Through CoG":
                ex1, ey1, ex2, ey2, length = self._cog_ray(mask, (px, py), (cx, cy))
            else:
                ex1, ey1, ex2, ey2, length = self._ray_cast(mask, (px, py), gx, gy, cont, grad_src)

            if length <= min_l:
                if length == 0: rejects['zero_length'] += 1
                else: rejects['too_short'] += 1
                continue
            if length >= max_l:
                rejects['too_long'] += 1
                continue

            dx, dy = ex1 - ex2, ey1 - ey2
            ang = np.arctan2(dy, dx)
            mx, my = (ex1 + ex2)/2, (ey1 + ey2)/2

            rank_score = length
            if algo == "Direct Line with CoG Boost":
                d2cog = np.sqrt((mx - cx)**2 + (my - cy)**2)
                prox = 1 - (d2cog / diag)
                rank_score = length - (boost * prox * 500)

            final_grasps.append(Candidate(
                x=mx, y=my, angle=ang, w=length,
                eq=t['eq'], dq=t['dq'], cq=t['cq'], comb=t['score'],
                l=rank_score
            ))

        final_grasps.sort(key=lambda x: x.line_length)

        info['num_valid'] = len(final_grasps)
        info['invalid_grasps'] = rejects
        info['has_contour'] = cont is not None
        viz_cands = [{'point': x['p'], 'combined_quality': x['score']} for x in top]
        info['top_candidates'] = viz_cands[:20] if len(viz_cands) >= 20 else viz_cands

        return final_grasps[:n_grasps], info


# ==========================================
# WRAPPER CLASS
# ==========================================

class AppWorkingGraspDetector:
    def __init__(self, edge_weight=0.001, depth_weight=0.001, cog_weight=0.999):
        self.detector = GraspDetector(edge_weight, depth_weight, cog_weight)

    def detect(self, rgb: np.ndarray, depth: np.ndarray, num_grasps: int = 10):
        grasps, info = self.detector.process(
            rgb=rgb,
            depth=depth,
            n_grasps=num_grasps,
            pct=DEPTH_PERCENTILE,
            mult=CANDIDATE_MULTIPLIER,
            min_l=MIN_GRASP_LENGTH,
            max_l=MAX_GRASP_LENGTH,
            algo=RAY_ALGORITHM,
            boost=COG_BOOST_VALUE,
            grad_src=GRADIENT_SOURCE,
            enable_crop=ENABLE_WHITE_CROP  # Use white cropping
        )
        return grasps, info


print("✓ Enhanced GraspDetector with white surface cropping loaded")
print("✓ Pipeline: Crop White Surface → Mask → Find COG → Detect Grasps")
