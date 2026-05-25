import numpy as np
import open3d as o3d
from scipy.ndimage import gaussian_filter
import cv2
import os
import yaml
import argparse


class HeatmapGenerator:
    def __init__(self, mesh_path: str, pose: np.ndarray, intrinsic: np.ndarray, scale: np.ndarray = None, use_avg: bool = False, gravity_direc: np.ndarray = [0,0,-1]):
        self.mesh = o3d.io.read_triangle_mesh(mesh_path)

        if scale is not None:
            vertices = np.array(self.mesh.vertices)

            self.mesh_center = vertices.mean(axis=0)
            vertices_centered = vertices - self.mesh_center #makes it (0, 0, 0)
            vertices_scaled = vertices_centered*scale
            vertices_recentered = vertices_scaled + self.mesh_center

            self.mesh.vertices = o3d.utility.Vector3dVector(vertices_recentered)
        self.pose = pose
        self.R = pose[:3, :3]
        self.T = pose[:3, 3]
        self.K = intrinsic

        self.gravity_direc = gravity_direc/np.linalg.norm(gravity_direc)
        self.mesh.compute_vertex_normals()
        print("***************")
        
        self.center_of_mass, self.cog_2d = self.compute_center_of_mass(use_avg)
        self.stability_scores = None
        self.scores_normalized = None
        self.top_points = {}

        print("HeatmapGenerator Class Initialised")

    def compute_center_of_mass(self, use_avg):
        vertices = np.array(self.mesh.vertices)
        if use_avg:
            center = np.mean(vertices, axis=0)
        else:
            min = vertices.min(axis=0)
            max = vertices.max(axis=0)
            center = (min+max)/2
        cog_camera_frame = self.R@center + self.T
        cog_calib = self.K@cog_camera_frame
        if(cog_calib[2]>0):
            u_cog = int(cog_calib[0]/cog_calib[2])
            v_cog = int(cog_calib[1]/cog_calib[2])
        return center, [u_cog, v_cog]
    
    def compute_stability_scores(self):
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

        vertical_score = np.dot(normals, - self.gravity_direc)
        is_top_facing = vertical_score > 0.7
        is_bottom_facing = vertical_score < -0.7
        is_side_facing = np.abs(vertical_score)<0.3

        height_norm = (vertices[:, 2] - min_bound[2])/(z_range + 1e-8)
        height_score = 1.0 - height_norm #closer to ground => more stable to grasp
        
        distance_from_cog = np.linalg.norm(vertices - self.center_of_mass, axis=1)
        cog_score = 1.0 - distance_from_cog/(distance_from_cog.max()+1e-8) #away from cog => less stable

        edge_distance = np.linalg.norm(vertices[:, :2] - self.center_of_mass[:2], axis=1)
        edge_score = 1.0 - edge_distance/(np.sqrt((x_range/2)**2 + (y_range/2)**2) + 1e-8) #closer to edge => easily graspable

        # Bottom surfaces
        stability_scores[is_bottom_facing] = (0.2*height_score[is_bottom_facing] + 0.3*cog_score[is_bottom_facing] + 
                                            0.3*edge_score[is_bottom_facing])
        # Side surfaces
        side_mask = is_side_facing & ~is_bottom_facing
        stability_scores[side_mask] = (0.6*cog_score[side_mask] + 0.6*edge_score[side_mask] + 
                                        0.65*(0.5 - np.abs(height_score[side_mask] - 0.5)))
        # Top surfaces
        top_mask = is_top_facing & ~is_bottom_facing & ~side_mask
        stability_scores[top_mask] = 0.5 * cog_score[top_mask] + 0.5*edge_score[top_mask]

        #other surfaces
        other_mask = ~(is_bottom_facing | is_side_facing | is_top_facing)
        stability_scores[other_mask] = 0.6 * cog_score[other_mask] + 0.5*edge_score[other_mask] + 0.2*height_score[other_mask]

        self.stability_scores = stability_scores
        s_min = np.min(self.stability_scores)
        s_max = np.max(self.stability_scores)
        self.scores_normalized = (stability_scores - s_min) / (s_max - s_min + 1e-8)
        
        return stability_scores

    def get_top_stable_points(self, color, num_points = 100):
        if self.stability_scores is None:
            scores = self.compute_stability_scores()
        scores = self.scores_normalized
        vertices = np.array(self.mesh.vertices)
        top_indices = np.argsort(scores)[-num_points:][::-1]

        print(len(top_indices))

        R = self.R
        T = self.T
        K = self.K

        for idx in top_indices:
            vertex = vertices[idx]
            vertex_camera_frame = R@vertex + T
            vertex_calib = K@vertex_camera_frame
            if(vertex_calib[2]<=0):
                continue
            u = int(vertex_calib[0]/vertex_calib[2])
            v = int(vertex_calib[1]/vertex_calib[2])
            if 0 <= u < color.shape[1] and 0 <= v < color.shape[0]:
                self.top_points[idx] = [u, v]
        
        print(len(self.top_points), "************")

    def generate_heatmap(self, color, intrinsic, pose, mask):
        (w, h) = color.shape[:2]
        heatmap = np.zeros((w, h), dtype=np.float32)

        stability_scores = self.compute_stability_scores()
        vertices = np.array(self.mesh.vertices)

        scores_norm = (stability_scores - stability_scores.min())/(stability_scores.max() - stability_scores.min() + 1e-8)
        top_idx = set(np.argsort(scores_norm)[-5:][::-1].tolist())
        print("Top 5 stable vertex indices:", top_idx)
        R = pose[:3, :3]
        T = pose[:3, 3].reshape((3, 1))
        # print(R.shape, T.shape, vertices.shape)
        vertices_homo = np.hstack([vertices, np.ones((len(vertices), 1))])
        # print(vertices_homo.shape)
        vertices_camera_frame = (pose @ vertices_homo.T).T[:, :3] #R@vertices.T + T
        vertices_calib = intrinsic@vertices_camera_frame.T
        vertices_calib = vertices_calib.T
        # print("calibrated vertices shape: ", vertices_calib)

        valid = vertices_calib[:, 2]>0 
        pixel_coords = vertices_calib[valid][:, :2]/vertices_calib[valid][:, 2:3]
        # print("********", pixel_coords[0:10])

        valid_scores = scores_norm[valid]

        for (u, v), score in zip(pixel_coords.astype(int), valid_scores):
            if 0 <= u < w and 0 <= v < h:
                cv2.circle(heatmap, (int(u), int(v)), 2, float(score), -1)

        sigma = 16
        heatmap_smooth = gaussian_filter(heatmap, sigma=sigma)

        mask_bool = mask > 0 if mask.dtype != bool else mask
        heatmap = heatmap_smooth * mask_bool

        heatmap = (heatmap - heatmap.min())/(heatmap.max() - heatmap.min() + 1e-8)
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        overlay = color.copy()
        
        overlay[mask_bool] = cv2.addWeighted(color[mask_bool], 0.4, heatmap_color[mask_bool], 0.6, 0)

        return heatmap, overlay
    
    def draw_top_stable_points(self, color, draw_cog=True):
        scores = self.scores_normalized
        image_draw = color.copy()
        colors = [(255,50,50), (255,165,0), (255,255,0), (50,255,50), (50,255,255)]
        
        if draw_cog:
            u_cog = self.cog_2d[0]
            v_cog = self.cog_2d[1]
            cv2.circle(image_draw, (u_cog, v_cog), 7, (0, 0, 0), 3)  # Black outline
            cv2.circle(image_draw, (u_cog, v_cog), 6, (255, 0, 255), -1)  # Magenta fill
            text = f"CoG"
            cv2.putText(image_draw, text, (u_cog+5, v_cog+15 ), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
                
        rank = 0        
        for idx, value in self.top_points.items():
            u, v = value
            
            c = colors[idx % len(colors)]
            # Draw with black outline for visibility on any background
            cv2.circle(image_draw, (u, v), 4, (0, 0, 0), 3)  # Black outline
            cv2.circle(image_draw, (u, v), 3, c, -1)  # color fill
            text = f"#{rank+1}: {scores[idx]:.2f}"
            cv2.circle(image_draw, (u, v), 4, (0, 0, 0), 3)
            cv2.circle(image_draw, (u, v), 3, c, -1)
            text = f"#{rank+1}: {scores[idx]:.2f}"
            cv2.putText(image_draw, text, (u+20, v+20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1)

            rank+=1
        
        return image_draw
    
    def compute_grasp_candidates(self, mask, num_candidates=10):  
        if self.scores_normalized is None:
            self.compute_stability_scores()
        
        scores = self.scores_normalized
        normals = np.array(self.mesh.vertex_normals)
        
        R = self.R
        
        grasp_candidates = []
        for idx, (u, v) in self.top_points.items():

            normal = R @ normals[idx]  # normal in camera frame
            p1, p2, grasp_angle = self.get_second_edge_point(mask, u, v, normal)

            width = np.linalg.norm(np.array(p1) - np.array(p2))
            cx, cy = (p1[0]+p2[0])//2, (p1[1]+p2[1])//2  # midpoint of the grasp region
            
            if width > 15:  
                grasp_candidates.append({
                    "x": cx, "y": cy,
                    "angle": grasp_angle,
                    "p1": p1, "p2": p2,
                    "width": width,
                    "score": scores[idx]
                })
        return grasp_candidates

    def cast_ray(self, mask, x, y, angle, max_dist):
        #find diagonally opposite point inside mask
        dx, dy = np.cos(angle), np.sin(angle)
        h, w = mask.shape
        
        ray = (x, y)
        for dist in range(1, max_dist):
            nx = int(x + dist * dx)
            ny = int(y + dist * dy)
            if not (0 <= ny < h and 0 <= nx < w):
                return ray
            
            if mask[ny, nx]:
                ray = (nx, ny)
            else:
                return ray
        
        return ray

    def compute_grasp_candidates_vertex(self, num_candidates=10):
        R = self.R
        T = self.T
        K = self.K
        scores = self.scores_normalized
        grasp_candidates = []
        for idx, (value) in self.top_points.items():
            score = scores[idx]
            u1, v1 = value
            
            # Find opposite vertex along normal direction
            opp_vertex = self.find_opposite_vertex(idx)
            
            if opp_vertex is None:
                continue
            
            # Project opposite vertex to image
            opp_cam = K @(R @ opp_vertex + T)
            if opp_cam[2] == 0:
                continue
            u2 = int(opp_cam[0] / opp_cam[2])
            v2 = int(opp_cam[1] / opp_cam[2])
            
            # Grasp center is midpoint
            center_x = (u1 + u2) / 2
            center_y = (v1 + v2) / 2
            
            # Width in pixels
            # width = np.sqrt((u2 - u1)**2 + (v2 - v1)**2)
            width = u2-u1
            
            # Angle from the two projected points
            grasp_angle = np.arctan2(v2 - v1, u2 - u1)
            
            if 10 < width < 400:
                grasp_candidates.append({
                    'x': int(center_x),
                    'y': int(center_y),
                    'angle': grasp_angle,
                    'width': width,
                    'score':score,
                    'p1': (u1, v1),
                    'p2': (u2, v2)
                })
            
            if len(grasp_candidates) >= num_candidates:
                break
        print("number of grasp candidates", len(grasp_candidates))
        return grasp_candidates
        
    
    def draw_grasp_candidates_vertex(self, color, grasp_candidates, top_n=10):
        image = color.copy()
        
        for i, grasp in enumerate(grasp_candidates[:top_n]):
            x, y = grasp['x'], grasp['y']
            angle = grasp['angle']
            width = grasp['width']
            p1 = grasp['p1']
            p2 = grasp['p2']
            
            height = 20
            
            # Color by rank
            color_val = (100, 255 - i*40, 100)
            
            rect = ((x, y), (width, height), np.degrees(angle))
            box = cv2.boxPoints(rect).astype(int)
            cv2.drawContours(image, [box], 0, color_val, 2)
            cv2.circle(image, (x, y), 5, color_val, -1)
            cv2.line(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 255, 0), 2)
            cv2.circle(image, (int(p1[0]), int(p1[1])), 4, (255, 0, 0), -1)  # Red
            cv2.circle(image, (int(p2[0]), int(p2[1])), 4, (0, 0, 255), -1)  # Blue
            cv2.putText(image, f"#{i+1}: {grasp['score']:.2f}", 
                        (x+15, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_val, 1)
        
        return image

    def get_second_edge_point(self, mask, u, v, normal_cam, max_dist=300):
        """Find the second grasp edge by ray casting PERPENDICULAR to normal."""
        nx, ny, _ = normal_cam

        grasp_angle = np.arctan2(ny, nx) #normal is already perpendicular to surface so taking normal projection on surface 
        return self.cast_ray(mask, u, v, grasp_angle, max_dist), self.cast_ray(mask, u, v, grasp_angle + np.pi, max_dist), grasp_angle

    def find_opposite_vertex(self, vertex_idx):
        #find opposite vertex in 3D
        vertices = np.array(self.mesh.vertices)
        normals = np.array(self.mesh.vertex_normals)
        
        v1 = vertices[vertex_idx]
        n1 = normals[vertex_idx]
        nx, ny, _ = n1
        
        tangent_dir = np.array([nx, ny, 0])
        tangent_dir = tangent_dir / (np.linalg.norm(tangent_dir) + 1e-8)
        
        best_opposite = None
        best_score = -1000
        
        for idx in range(len(vertices)):
            if idx == vertex_idx:
                continue
            
            v2 = vertices[idx]
            
            # Height difference - prefer same height
            height_diff = abs(v2[2] - v1[2])
            
            # Vector from v1 to v2
            v1_to_v2 = v2 - v1
            dist = np.linalg.norm(v1_to_v2)
            
            if dist < 1e-6:
                continue
            
            v1_to_v2_norm = v1_to_v2 / dist
            
            # Check alignment with tangent direction (+ or - direction)
            alignment = abs(np.dot(v1_to_v2_norm, tangent_dir))
            
            # Filter: good alignment and similar height
            bbox = self.mesh.get_axis_aligned_bounding_box()
            z_range = bbox.get_max_bound()[2] - bbox.get_min_bound()[2]
            max_height_diff = z_range * 0.1
            
            if alignment > 0.5 and height_diff < max_height_diff:
                # Score: prefer high alignment, low height diff, reasonable distance
                score = alignment - height_diff * 5.0
                
                if score > best_score:
                    best_score = score
                    best_opposite = v2
        
        return best_opposite
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, default="red_bottle", help="Object name")
    parser.add_argument("--use_vertex", action="store_true", help="Use vertices for ray casting")
    parser.add_argument("--use_cog_avg", action="store_true", help="Use average of all vertices as center")
    args = parser.parse_args()
    
    obj = args.obj
    demo_path = f"demo_data_{obj}"
    # demo_path = "demo_data_midas"
    results_path = f"results/demo_{obj}"
    # results_path = "results/demo_demo_mustard_full_data"
    
    mesh_path = f"{results_path}/final_mesh_{obj}.obj"
    
    mesh_scale_path = os.path.join(results_path, f"{obj}_mesh_scale.npy")
    metadata_path = os.path.join(results_path, f"{obj}_resize_metadata.npy")

    if os.path.exists(metadata_path):
        metadata = np.load(metadata_path, allow_pickle=True).item()
        image_scale = metadata.get("image_scale", 1.0)
        resize = metadata['resized_size']
        print("Loaded metadata:", metadata)
        
    else:
        resize = None
    
    if os.path.exists(mesh_scale_path):
        mesh_scale = np.load(mesh_scale_path)
    else:
        mesh_scale = [1.0, 1.0, 1.0]

    poses = np.load(os.path.join(results_path, f"{obj}_poses.npy"))
    scores = np.load(os.path.join(results_path, f"{obj}_scores.npy"))
    best_score_idx = np.argmax(scores)
    best_pose = poses[best_score_idx]
    # label = np.load(os.path.join(demo_path, 'labels.npz'))
    # obj_num = 5
    # mask = np.where(label['seg'] == obj_num, 255, 0).astype(np.bool_)

    mask_file = np.load(os.path.join(demo_path, "labels_filtered.npz"), allow_pickle=True)
    # mask = mask_file[5]['seg']
    mask = mask_file['arr_0'].item()['segmentation'].astype(np.bool_)
    color_original = cv2.cvtColor(cv2.imread(os.path.join(demo_path, 'color.png')), cv2.COLOR_BGR2RGB)
    
    intrinsic_path = os.path.join(demo_path, "836212060125_640x480.yml")
    with open(intrinsic_path, 'r') as f:
        intrinsic_data = yaml.load(f, Loader=yaml.FullLoader)
    K = np.array([[intrinsic_data['color']['fx'], 0.0, intrinsic_data['color']['ppx']],
                  [0.0, intrinsic_data['color']['fy'], intrinsic_data['color']['ppy']],
                  [0.0, 0.0, 1.0]])
    if resize is not None:
        K[0, 0] *= image_scale
        K[1, 1] *= image_scale
        K[0, 2] *= image_scale
        K[1, 2] *= image_scale
        new_w, new_h = resize
        color = cv2.resize(color_original, (new_w, new_h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        print("Resized images to:", f"({new_w}, {new_h})")
    else:
        color = color_original
        
    generator = HeatmapGenerator(mesh_path, best_pose, K, scale=mesh_scale, use_avg = args.use_cog_avg)
    _, overlay = generator.generate_heatmap(color,K, best_pose, mask)
    n_points = 100
    generator.get_top_stable_points(color, n_points)

    best_points_image_heatmap = generator.draw_top_stable_points(overlay, draw_cog=True)
    best_points_image = generator.draw_top_stable_points(color, draw_cog=True)
    
    if args.use_vertex:
        grasp_candidates = generator.compute_grasp_candidates_vertex()
        print("Number of grasp candidates found: ", len(grasp_candidates))
        grasp_image = generator.draw_grasp_candidates_vertex(overlay, grasp_candidates)
    else:
        grasp_candidates = generator.compute_grasp_candidates(mask)
        print("Number of grasp candidates found: ", len(grasp_candidates))
        grasp_image = generator.draw_grasp_candidates_vertex(overlay, grasp_candidates)

    output_dir = f'stability_results_{obj}'
    print("Saving results to:", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, f'{obj}_heatmap_overlay.png'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, f'{obj}_annotated.png'), cv2.cvtColor(best_points_image_heatmap, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, f'{obj}_best_{n_points}_points.png'), cv2.cvtColor(best_points_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, f'{obj}_best_{n_points}_grasp.png'), cv2.cvtColor(grasp_image, cv2.COLOR_RGB2BGR))
    
if __name__ == "__main__":
    main()

