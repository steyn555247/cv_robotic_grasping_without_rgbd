import numpy as np
import open3d as o3d
from scipy.ndimage import gaussian_filter
import cv2
import os
import yaml
import tempfile
import base64
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='static')
CORS(app)

UPLOAD_FOLDER = tempfile.mkdtemp()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


class HeatmapGeneratorConfigurable:
    def __init__(self, mesh_path: str, pose: np.ndarray, intrinsic: np.ndarray, 
                 scale: np.ndarray = None, use_avg: bool = False, gravity_direc: np.ndarray = None):
        if gravity_direc is None:
            gravity_direc = [0, 0, -1]
        
        self.mesh = o3d.io.read_triangle_mesh(mesh_path)

        if scale is not None:
            vertices = np.array(self.mesh.vertices)
            self.mesh_center = vertices.mean(axis=0)
            vertices_centered = vertices - self.mesh_center
            vertices_scaled = vertices_centered * scale
            vertices_recentered = vertices_scaled + self.mesh_center
            self.mesh.vertices = o3d.utility.Vector3dVector(vertices_recentered)

        self.pose = pose
        self.R = pose[:3, :3]
        self.T = pose[:3, 3]
        self.K = intrinsic

        self.gravity_direc = np.array(gravity_direc) / np.linalg.norm(gravity_direc)
        self.mesh.compute_vertex_normals()

        self.center_of_mass, self.cog_2d = self.compute_center_of_mass(use_avg)
        self.stability_scores = None
        self.scores_normalized = None
        self.top_points = {}

    def compute_center_of_mass(self, use_avg):
        vertices = np.array(self.mesh.vertices)
        if use_avg:
            center = np.mean(vertices, axis=0)
        else:
            min_v = vertices.min(axis=0)
            max_v = vertices.max(axis=0)
            center = (min_v + max_v) / 2
        cog_camera_frame = self.R @ center + self.T
        cog_calib = self.K @ cog_camera_frame
        u_cog, v_cog = 0, 0
        if cog_calib[2] > 0:
            u_cog = int(cog_calib[0] / cog_calib[2])
            v_cog = int(cog_calib[1] / cog_calib[2])
        return center, [u_cog, v_cog]

    def compute_stability_scores(self, params):
        """Compute stability scores with configurable parameters."""
        vertices = np.array(self.mesh.vertices)
        normals = np.array(self.mesh.vertex_normals)
        num_vertices = vertices.shape[0]
        stability_scores = np.zeros(num_vertices, dtype=np.float32)

        bbox = self.mesh.get_axis_aligned_bounding_box()
        min_bound = bbox.get_min_bound()
        max_bound = bbox.get_max_bound()

        x_range = max_bound[0] - min_bound[0]
        y_range = max_bound[1] - min_bound[1]
        z_range = max_bound[2] - min_bound[2]

        vertical_score = np.dot(normals, -self.gravity_direc)
        
        # Configurable thresholds
        top_threshold = params.get('top_threshold', 0.7)
        bottom_threshold = params.get('bottom_threshold', -0.7)
        side_threshold = params.get('side_threshold', 0.3)
        
        is_top_facing = vertical_score > top_threshold
        is_bottom_facing = vertical_score < bottom_threshold
        is_side_facing = np.abs(vertical_score) < side_threshold

        height_norm = (vertices[:, 2] - min_bound[2]) / (z_range + 1e-8)
        height_score = 1.0 - height_norm

        distance_from_cog = np.linalg.norm(vertices - self.center_of_mass, axis=1)
        cog_score = 1.0 - distance_from_cog / (distance_from_cog.max() + 1e-8)

        edge_distance = np.linalg.norm(vertices[:, :2] - self.center_of_mass[:2], axis=1)
        edge_score = 1.0 - edge_distance / (np.sqrt((x_range/2)**2 + (y_range/2)**2) + 1e-8)

        # Configurable weights for bottom surfaces
        bottom_height_w = params.get('bottom_height_weight', 0.2)
        bottom_cog_w = params.get('bottom_cog_weight', 0.3)
        # bottom_vertical_w = params.get('bottom_vertical_weight', 0.2)
        bottom_edge_w = params.get('bottom_edge_weight', 0.3)
        
        stability_scores[is_bottom_facing] = (
            bottom_height_w * height_score[is_bottom_facing] +
            bottom_cog_w * cog_score[is_bottom_facing] +
            bottom_edge_w * edge_score[is_bottom_facing]
        )

        # Configurable weights for side surfaces
        side_cog_w = params.get('side_cog_weight', 0.4)
        side_edge_w = params.get('side_edge_weight', 0.3)
        side_height_w = params.get('side_height_weight', 0.3)
        
        side_mask = is_side_facing & ~is_bottom_facing
        stability_scores[side_mask] = (
            side_cog_w * cog_score[side_mask] +
            side_edge_w * edge_score[side_mask] +
            side_height_w * (0.5 - np.abs(height_score[side_mask] - 0.5))
        )

        # Configurable weights for top surfaces
        top_cog_w = params.get('top_cog_weight', 0.2)
        top_edge_w = params.get('top_edge_weight', 0.3)
        
        top_mask = is_top_facing & ~is_bottom_facing & ~side_mask
        stability_scores[top_mask] = (
            top_cog_w * cog_score[top_mask] +
            top_edge_w * edge_score[top_mask]
        )

        # Configurable weights for other surfaces
        other_cog_w = params.get('other_cog_weight', 0.3)
        other_edge_w = params.get('other_edge_weight', 0.3)
        other_height_w = params.get('other_height_weight', 0.2)
        
        other_mask = ~(is_bottom_facing | is_side_facing | is_top_facing)
        stability_scores[other_mask] = (
            other_cog_w * cog_score[other_mask] +
            other_edge_w * edge_score[other_mask] +
            other_height_w * height_score[other_mask]
        )

        self.stability_scores = stability_scores
        s_min = np.min(self.stability_scores)
        s_max = np.max(self.stability_scores)
        self.scores_normalized = (stability_scores - s_min) / (s_max - s_min + 1e-8)

        return stability_scores

    def get_top_stable_points(self, color, num_points=100):
        """Get top stable points - matches original: get_top_stable_points(color, best_pose, K, 100)"""
        if self.scores_normalized is None:
            return
        scores = self.scores_normalized
        vertices = np.array(self.mesh.vertices)
        top_indices = np.argsort(scores)[-num_points:][::-1]

        self.top_points = {}
        for idx in top_indices:
            vertex = vertices[idx]
            vertex_camera_frame = self.R @ vertex + self.T
            vertex_calib = self.K @ vertex_camera_frame
            if vertex_calib[2] <= 0:
                continue
            u = int(vertex_calib[0] / vertex_calib[2])
            v = int(vertex_calib[1] / vertex_calib[2])
            if 0 <= u < color.shape[1] and 0 <= v < color.shape[0]:
                self.top_points[idx] = [u, v]

    def generate_heatmap(self, color, mask, params):
        (w, h) = color.shape[:2]
        heatmap = np.zeros((w, h), dtype=np.float32)

        stability_scores = self.compute_stability_scores(params)
        vertices = np.array(self.mesh.vertices)

        scores_norm = self.scores_normalized

        vertices_homo = np.hstack([vertices, np.ones((len(vertices), 1))])
        vertices_camera_frame = (self.pose @ vertices_homo.T).T[:, :3]
        vertices_calib = (self.K @ vertices_camera_frame.T).T

        valid = vertices_calib[:, 2] > 0
        pixel_coords = vertices_calib[valid][:, :2] / vertices_calib[valid][:, 2:3]
        valid_scores = scores_norm[valid]

        for (u, v), score in zip(pixel_coords.astype(int), valid_scores):
            if 0 <= u < h and 0 <= v < w:
                cv2.circle(heatmap, (int(u), int(v)), 2, float(score), -1)

        sigma = params.get('gaussian_sigma', 16)
        heatmap_smooth = gaussian_filter(heatmap, sigma=sigma)

        mask_bool = mask > 0 if mask.dtype != bool else mask
        heatmap = heatmap_smooth * mask_bool

        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        overlay = color.copy()
        overlay[mask_bool] = cv2.addWeighted(
            color[mask_bool], 0.4, heatmap_color[mask_bool], 0.6, 0
        )

        return heatmap, overlay

    def draw_top_stable_points(self, color, draw_cog=True, max_display=5):
        """Draw top stable points with legend panel - matches original"""
        scores = self.scores_normalized
        image_draw = color.copy()
        colors = [(255, 50, 50), (255, 165, 0), (255, 255, 0), (50, 255, 50), (50, 255, 255)]
        
        panel_width = 240
        panel = np.zeros((image_draw.shape[0], panel_width, 3), dtype=np.uint8) + 30

        y0 = 40
        dy = 30

        cv2.putText(panel, "Stability Ranking", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if draw_cog:
            u_cog, v_cog = self.cog_2d
            cv2.circle(image_draw, (u_cog, v_cog), 7, (0, 0, 0), 3)
            cv2.circle(image_draw, (u_cog, v_cog), 6, (255, 0, 255), -1)
            cv2.putText(image_draw, "CoG", (u_cog + 5, v_cog + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        rank = 0
        for idx, value in self.top_points.items():
            if rank >= max_display:
                break
            u, v = value
            c = colors[rank % len(colors)]
            cv2.circle(image_draw, (u, v), 4, (0, 0, 0), 3)
            cv2.circle(image_draw, (u, v), 3, c, -1)
            text = f"#{rank+1}: {scores[idx]:.2f}"
            cv2.putText(image_draw, text, (u + 20, v + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1)
            rank += 1

        image_draw = np.hstack([image_draw, panel])
        return image_draw

    def _cast_ray(self, mask, x, y, angle, max_dist):
        """Cast ray and return the LAST point that is INSIDE the mask."""
        dx, dy = np.cos(angle), np.sin(angle)
        h, w = mask.shape
        
        last_inside = (x, y)
        
        for dist in range(1, max_dist):
            nx = int(x + dist * dx)
            ny = int(y + dist * dy)
            
            if not (0 <= ny < h and 0 <= nx < w):
                return last_inside
            
            if mask[ny, nx]:
                last_inside = (nx, ny)
            else:
                return last_inside
        
        return last_inside

    def _find_second_edge_point(self, mask, u, v, normal_cam, max_dist=300):
        """Find the second grasp edge by ray casting PERPENDICULAR to normal."""
        nx, ny, nz = normal_cam
        grasp_angle = np.arctan2(ny, nx) #+ np.pi/2

        return (self._cast_ray(mask, u, v, grasp_angle, max_dist),
                self._cast_ray(mask, u, v, grasp_angle + np.pi, max_dist),
                grasp_angle)

    def compute_grasp_candidates(self, mask, num_candidates=10):
        """Compute grasp candidates using mask-based ray casting (use_vertex=False)"""
        if self.scores_normalized is None:
            self.compute_stability_scores({})
        
        scores = self.scores_normalized
        normals = np.array(self.mesh.vertex_normals)
        
        mask_bool = mask > 0 if mask.dtype != bool else mask
        
        grasp_candidates = []
        for idx, (u, v) in self.top_points.items():
            normal_cam = self.R @ normals[idx]
            
            p1, p2, grasp_angle = self._find_second_edge_point(mask_bool, u, v, normal_cam)
            
            width = np.linalg.norm(np.array(p1) - np.array(p2))
            cx, cy = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
            
            if width > 15:
                grasp_candidates.append({
                    "x": cx, "y": cy,
                    "angle": grasp_angle,
                    "p1": p1, "p2": p2,
                    "width": width,
                    "score": scores[idx]
                })
        
        return grasp_candidates

    def _find_opposite_vertex(self, vertex_idx):
        """Find vertex by traveling perpendicular to normal in XY plane."""
        vertices = np.array(self.mesh.vertices)
        normals = np.array(self.mesh.vertex_normals)
        
        v1 = vertices[vertex_idx]
        n1 = normals[vertex_idx]
        
        nx, ny, nz = n1
        
        tangent_dir = np.array([nx, ny, 0])
        tangent_dir = tangent_dir / (np.linalg.norm(tangent_dir) + 1e-8)
        
        best_opposite = None
        best_score = -np.inf
        
        v1_z = v1[2]
        
        for idx in range(len(vertices)):
            if idx == vertex_idx:
                continue
            
            v2 = vertices[idx]
            height_diff = abs(v2[2] - v1_z)
            
            v1_to_v2 = v2 - v1
            dist = np.linalg.norm(v1_to_v2)
            
            if dist < 1e-6:
                continue
            
            v1_to_v2_norm = v1_to_v2 / dist
            alignment = abs(np.dot(v1_to_v2_norm, tangent_dir))
            
            bbox = self.mesh.get_axis_aligned_bounding_box()
            z_range = bbox.get_max_bound()[2] - bbox.get_min_bound()[2]
            max_height_diff = z_range * 0.1
            
            if alignment > 0.5 and height_diff < max_height_diff:
                score = alignment - height_diff * 5.0
                
                if score > best_score:
                    best_score = score
                    best_opposite = (idx, v2, dist)
        
        return best_opposite

    def compute_grasp_candidates_vertex(self, num_candidates=10):
        """Compute grasp candidates using vertex-based method (use_vertex=True)"""
        normals = np.array(self.mesh.vertex_normals)
        scores = self.scores_normalized
        R = self.R
        T = self.T
        K = self.K

        grasp_candidates = []
        for idx, value in self.top_points.items():
            normal = normals[idx]
            score = scores[idx]
            u1, v1 = value
            
            opposite = self._find_opposite_vertex(idx)
            
            if opposite is None:
                continue
            
            opp_idx, opp_vertex, dist_3d = opposite
            
            opp_cam = R @ opp_vertex + T
            if opp_cam[2] <= 0:
                continue
            opp_proj = K @ opp_cam
            u2 = int(opp_proj[0] / opp_proj[2])
            v2 = int(opp_proj[1] / opp_proj[2])
            
            center_x = (u1 + u2) / 2
            center_y = (v1 + v2) / 2
            
            width = np.sqrt((u2 - u1)**2 + (v2 - v1)**2)
            grasp_angle = np.arctan2(v2 - v1, u2 - u1)
            
            if 10 < width < 400:
                grasp_candidates.append({
                    'x': int(center_x),
                    'y': int(center_y),
                    'angle': grasp_angle,
                    'width': width,
                    'width_3d': dist_3d,
                    'score': score,
                    'vertex_idx': idx,
                    'opposite_idx': opp_idx,
                    'p1': (u1, v1),
                    'p2': (u2, v2)
                })
            
            if len(grasp_candidates) >= num_candidates:
                break
        
        return grasp_candidates

    def draw_grasp_candidates_vertex(self, color, grasp_candidates, top_n=10):
        """Draw grasp rectangles - matches original"""
        image = color.copy()
        
        for i, grasp in enumerate(grasp_candidates[:top_n]):
            x, y = grasp['x'], grasp['y']
            angle = grasp['angle']
            width = grasp['width']
            p1 = grasp['p1']
            p2 = grasp['p2']
            
            height = 20
            
            color_val = (100, 255 - i*40, 100)
            
            rect = ((x, y), (width, height), np.degrees(angle))
            box = cv2.boxPoints(rect).astype(int)
            cv2.drawContours(image, [box], 0, color_val, 2)
            
            cv2.circle(image, (x, y), 5, color_val, -1)
            cv2.line(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 255, 0), 2)
            cv2.circle(image, (int(p1[0]), int(p1[1])), 4, (255, 0, 0), -1)
            cv2.circle(image, (int(p2[0]), int(p2[1])), 4, (0, 0, 255), -1)
            
            cv2.putText(image, f"#{i+1}: {grasp['score']:.2f}",
                        (x+15, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_val, 1)
        
        return image


# Global state to store uploaded files
uploaded_data = {}


def load_intrinsics_from_yml(yml_path):
    with open(yml_path, 'r') as f:
        intrinsic_data = yaml.load(f, Loader=yaml.FullLoader)
    K = np.array([
        [intrinsic_data['color']['fx'], 0.0, intrinsic_data['color']['ppx']],
        [0.0, intrinsic_data['color']['fy'], intrinsic_data['color']['ppy']],
        [0.0, 0.0, 1.0]
    ])
    return K


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    global uploaded_data
    
    try:
        files_info = {}
        
        file_keys = ['mesh', 'color', 'mask', 'pose', 'intrinsics', 'mesh_scale', 
                     'scores', 'resize_metadata']
        
        for key in file_keys:
            if key in request.files:
                f = request.files[key]
                if f.filename:
                    path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
                    f.save(path)
                    files_info[key] = path

        uploaded_data.update(files_info)
        
        return jsonify({'status': 'success', 'files': list(files_info.keys())})
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/generate', methods=['POST'])
def generate_heatmap():
    global uploaded_data
    
    try:
        params = request.json
        
        # Check required files
        required = ['mesh', 'color', 'mask', 'pose', 'intrinsics']
        missing = [f for f in required if f not in uploaded_data]
        if missing:
            return jsonify({'status': 'error', 'message': f'Missing files: {missing}'}), 400
        
        # Get boolean options (default False)
        use_cog_avg = params.get('use_cog_avg', False)
        use_vertex = params.get('use_vertex', False)
        
        # Load resize metadata if available
        image_scale = 1.0
        resize = None
        if 'resize_metadata' in uploaded_data:
            metadata = np.load(uploaded_data['resize_metadata'], allow_pickle=True).item()
            image_scale = metadata.get('image_scale', 1.0)
            resize = metadata.get('resized_size', None)
            print(f"Loaded resize metadata: scale={image_scale}, resize={resize}")
        
        # Load color image
        color_original = cv2.cvtColor(cv2.imread(uploaded_data['color']), cv2.COLOR_BGR2RGB)
        
        # Load mask - handle different formats
        mask_path = uploaded_data['mask']
        if mask_path.endswith('.npz'):
            mask_data = np.load(mask_path, allow_pickle=True)
            if 'arr_0' in mask_data:
                arr = mask_data['arr_0']
                if hasattr(arr, 'item'):
                    arr = arr.item()
                if isinstance(arr, dict) and 'segmentation' in arr:
                    mask_original = arr['segmentation'].astype(np.bool_)
                else:
                    mask_original = arr.astype(np.bool_)
            elif 'seg' in mask_data:
                mask_original = mask_data['seg'].astype(np.bool_)
            else:
                mask_original = list(mask_data.values())[0].astype(np.bool_)
        elif mask_path.endswith('.npy'):
            mask_original = np.load(mask_path).astype(np.bool_)
        else:
            mask_original = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0
        
        # Load poses and scores to find best pose
        poses = np.load(uploaded_data['pose'])
        
        if 'scores' in uploaded_data:
            scores = np.load(uploaded_data['scores'])
            best_score_idx = np.argmax(scores)
            pose = poses[best_score_idx]
            print(f"Using best pose index {best_score_idx} with score {scores[best_score_idx]}")
        elif poses.ndim == 3:
            pose = poses[0]
            print("No scores file, using first pose")
        else:
            pose = poses
        
        # Load intrinsics
        intrinsics_path = uploaded_data['intrinsics']
        if intrinsics_path.endswith('.yml') or intrinsics_path.endswith('.yaml'):
            K = load_intrinsics_from_yml(intrinsics_path)
        else:
            K = np.load(intrinsics_path)
        
        # Apply resize if metadata exists
        if resize is not None:
            K[0, 0] *= image_scale
            K[1, 1] *= image_scale
            K[0, 2] *= image_scale
            K[1, 2] *= image_scale
            new_w, new_h = resize
            color = cv2.resize(color_original, (new_w, new_h), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask_original.astype(np.uint8), (new_w, new_h), 
                            interpolation=cv2.INTER_NEAREST).astype(bool)
            print(f"Resized to ({new_w}, {new_h})")
        else:
            color = color_original
            mask = mask_original
        
        # Load mesh scale if available
        mesh_scale = None
        if 'mesh_scale' in uploaded_data:
            mesh_scale = np.load(uploaded_data['mesh_scale'])
        
        # Create generator with use_avg option
        generator = HeatmapGeneratorConfigurable(
            uploaded_data['mesh'], pose, K, scale=mesh_scale, use_avg=use_cog_avg
        )
        
        # Generate heatmap with params
        heatmap, overlay = generator.generate_heatmap(color, mask, params)
        
        # Get top stable points (use 100 internally like original, display fewer)
        generator.get_top_stable_points(color, num_points=100)
        
        # Get display count from params
        num_display = int(params.get('num_points', 5))
        
        # Draw points on overlay with legend
        points_image = generator.draw_top_stable_points(overlay, draw_cog=True, max_display=num_display)
        
        # Compute grasp candidates based on use_vertex option
        if use_vertex:
            grasp_candidates = generator.compute_grasp_candidates_vertex(num_candidates=num_display)
            print(f"Using vertex-based grasp candidates: {len(grasp_candidates)}")
        else:
            grasp_candidates = generator.compute_grasp_candidates(mask, num_candidates=num_display)
            print(f"Using mask-based grasp candidates: {len(grasp_candidates)}")
        
        # Extract point data for display
        vertices = np.array(generator.mesh.vertices)
        point_data = []
        
        point_data.append({
            "rank": "CoG",
            "vertex_idx": "—",
            "pixel": [generator.cog_2d[0], generator.cog_2d[1]],
            "pose_3d": generator.center_of_mass.tolist(),
            "score": float(1.0)
        })
        count = 0
        for idx, (u, v) in generator.top_points.items():
            if count >= num_display:
                break
            point_data.append({
                "rank": count + 1,
                "vertex_idx": int(idx),
                "pixel": [u, v],
                "pose_3d": vertices[idx].tolist(),
                "score": float(generator.scores_normalized[idx])
            })
            count += 1
        
        grasp_image = generator.draw_grasp_candidates_vertex(overlay, grasp_candidates, top_n=num_display)
        
        # Encode both images as base64
        _, points_buffer = cv2.imencode('.png', cv2.cvtColor(points_image, cv2.COLOR_RGB2BGR))
        points_base64 = base64.b64encode(points_buffer).decode('utf-8')
        
        _, grasp_buffer = cv2.imencode('.png', cv2.cvtColor(grasp_image, cv2.COLOR_RGB2BGR))
        grasp_base64 = base64.b64encode(grasp_buffer).decode('utf-8')
        
        # Also return heatmap overlay only
        _, heatmap_buffer = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        heatmap_base64 = base64.b64encode(heatmap_buffer).decode('utf-8')

        return jsonify({
            'status': 'success',
            'points_image': points_base64,
            'grasp_image': grasp_base64,
            'heatmap': heatmap_base64,
            'point_data': point_data
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    print("Starting Heatmap Demo Server...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)
