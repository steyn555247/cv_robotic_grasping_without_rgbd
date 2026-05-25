import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import gc
import time
# from dataclasses import dataclass # didn't need this actually
import plotly.graph_objects as go
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# ==========================================
# CLASSES
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
        
        # local corners
        pts = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
        
        res = []
        for dx, dy in pts:
            px = self.x + dx * c - dy * s
            py = self.y + dx * s + dy * c
            res.append([px, py])
            
        return np.array(res)

class GraspDetector:
    def __init__(self, w_e, w_d, w_c):
        self.we = w_e
        self.wd = w_d
        self.wc = w_c

    def _ray_cast(self, mask, start, vec_x, vec_y, contour=None, mode="Depth Gradients", limit=500):
        # This handles the ray casting logic. 
        # Supports multiple modes for direction finding.
        h, w = mask.shape
        gx, gy = start
        
        # Determine direction (dx, dy)
        dx, dy = 1.0, 0.0 # defaults

        if mode == "Contour Direction (80px avg)":
            if contour is not None:
                pts = contour.reshape(-1, 2)
                
                # get nearest point index
                dists = np.sqrt((pts[:, 0] - gx)**2 + (pts[:, 1] - gy)**2)
                idx = np.argmin(dists)
                
                # grab a segment ~80px
                n_pts = min(40, len(pts) // 4)
                half = n_pts // 2
                
                s_idx = (idx - half) % len(pts)
                e_idx = (idx + half) % len(pts)
                
                if s_idx < e_idx:
                    seg = pts[s_idx:e_idx+1]
                else:
                    seg = np.vstack([pts[s_idx:], pts[:e_idx+1]])
                
                if len(seg) > 2:
                    # PCA approach for tangent
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
                            # rotate 90 deg for normal
                            dx = -ty
                            dy = tx
                        else:
                            dx, dy = 1.0, 0.0
                    else:
                        dx, dy = 1.0, 0.0
                else:
                    # not enough points, simple diff
                    if len(seg) >= 2:
                        tx = seg[-1, 0] - seg[0, 0]
                        ty = seg[-1, 1] - seg[0, 1]
                        l = np.sqrt(tx**2 + ty**2)
                        if l > 1e-6:
                            tx /= l; ty /= l
                            dx, dy = -ty, tx
        
        elif mode == "Depth Gradients":
            # use precalculated gradients passed as args
            if 0 <= gy < vec_y.shape[0] and 0 <= gx < vec_x.shape[1]:
                val_x = vec_x[gy, gx]
                val_y = vec_y[gy, gx]
            else:
                # bounds check failed, sample area
                ks = 3
                val_x = np.mean(vec_x[max(0, gy-ks):min(h, gy+ks+1), max(0, gx-ks):min(w, gx+ks+1)])
                val_y = np.mean(vec_y[max(0, gy-ks):min(h, gy+ks+1), max(0, gx-ks):min(w, gx+ks+1)])
            
            lenght = np.sqrt(val_x**2 + val_y**2)
            if lenght < 1e-6:
                # radial fallback
                dx = gx - w // 2
                dy = gy - h // 2
                lenght = np.sqrt(dx**2 + dy**2)
                if lenght > 1e-6: dx /= lenght; dy /= lenght
            else:
                # perpendicular
                dx = -val_y / lenght
                dy = val_x / lenght

        elif mode == "Image Edges":
             # similar to depth but on rgb edges
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

        # Trace both ways
        x1, y1 = self._trace_line(mask, gx, gy, dx, dy, limit)
        x2, y2 = self._trace_line(mask, gx, gy, -dx, -dy, limit)
        
        full_dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        return x1, y1, x2, y2, full_dist

    def _cog_ray(self, mask, gp, cog, limit=500):
        gx, gy = gp
        cx, cy = cog
        
        # Vector from COG to Grasp Point
        vx = gx - cx
        vy = gy - cy
        
        dist = np.sqrt(vx**2 + vy**2)
        if dist < 1e-6: return cx, cy, cx, cy, 0.0
        
        vx /= dist
        vy /= dist
        
        # Forward and back
        e1x, e1y = self._trace_line(mask, cx, cy, vx, vy, limit)
        e2x, e2y = self._trace_line(mask, cx, cy, -vx, -vy, limit)
        
        l = np.sqrt((e1x-e2x)**2 + (e1y-e2y)**2)
        return e1x, e1y, e2x, e2y, l

    def _trace_line(self, mask, sx, sy, dx, dy, limit):
        h, w = mask.shape
        cx, cy = float(sx), float(sy)
        
        # Allow small gaps (noise tolerance)
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
        # 1. Depth threshold
        thresh = np.percentile(dmap, pct)
        droi = (dmap >= thresh).astype(np.uint8) * 255
        
        k1 = np.ones((7,7), np.uint8)
        droi = cv2.morphologyEx(droi, cv2.MORPH_CLOSE, k1)
        droi = cv2.morphologyEx(droi, cv2.MORPH_OPEN, k1)

        # 2. Canny
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 100)

        # 3. Combine
        e_roi = cv2.bitwise_and(edges, edges, mask=droi)
        dilated = cv2.dilate(e_roi, np.ones((5,5), np.uint8), iterations=3)

        blend = (droi.astype(np.float32) * 0.2 + dilated.astype(np.float32) * 0.8).astype(np.uint8)
        _, bin_mask = cv2.threshold(blend, 100, 255, cv2.THRESH_BINARY)

        # 4. Clean
        final_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

        # 5. Contours
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

        # 6. Grads
        d8 = (dmap * 255).astype(np.uint8)
        gx = cv2.Sobel(d8, cv2.CV_64F, 1, 0, ksize=5)
        gy = cv2.Sobel(d8, cv2.CV_64F, 0, 1, ksize=5)
        
        return final_mask, main_c, gx, gy, e_roi

    def process(self, rgb, depth, n_grasps=10, pct=60, mult=3, min_l=100, max_l=1000, 
                algo="Through CoG", boost=0.0, grad_src="Depth Gradients"):
        
        h, w = rgb.shape[:2]
        info = {}
        
        mask, cont, gx, gy, ed = self.get_mask_data(rgb, depth, pct)
        
        # normalize depth grad
        mag = np.sqrt(gx**2 + gy**2)
        norm_g = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
        
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        enorm = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0

        # COG
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
        
        # Sample points
        if cont is not None:
            pts = cont.reshape(-1, 2)
            # take 100 points roughly
            idx = np.linspace(0, len(pts)-1, min(len(pts), 100), dtype=int)
            cands = pts[idx]
        else:
            cands = [[w//2, h//2]]
            
        info['num_contour_points'] = len(cands)

        # Rank candidates
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
        
        final_grasps = []
        rejects = {'too_short': 0, 'too_long': 0, 'zero_length': 0}
        
        # Ray cast logic
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
            
        # sort by smallest length (or boosted score)
        final_grasps.sort(key=lambda x: x.line_length)
        
        info['num_valid'] = len(final_grasps)
        info['invalid_grasps'] = rejects
        info['has_contour'] = cont is not None
        # remap keys for debug viz
        viz_cands = [{'point': x['p'], 'combined_quality': x['score']} for x in top]
        info['top_candidates'] = viz_cands[:20] if len(viz_cands) >= 20 else viz_cands
        
        return final_grasps[:n_grasps], info

# ==========================================
# UTILS & MODELS
# ==========================================

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def clear_memory(new_name):
    if 'model_name' in st.session_state and st.session_state.model_name != new_name:
        st.toast(f"Switching to {new_name}...", icon="🧹")
        if 'depth_wrapper' in st.session_state:
            del st.session_state.depth_wrapper
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            st.session_state.model_loaded = False
    st.session_state.model_name = new_name

class DepthWrapper:
    def __init__(self, label, device):
        self.dev = device
        self.label = label
        self.proc = None
        self.model = None
        self.tf = None
        
        # map config
        MAP = {
            "DepthAnythingV2-Small": {"t": "da", "id": "depth-anything/Depth-Anything-V2-Small-hf"},
            "DepthAnythingV2-Base":  {"t": "da", "id": "depth-anything/Depth-Anything-V2-Base-hf"},
            "DepthAnythingV2-Large": {"t": "da", "id": "depth-anything/Depth-Anything-V2-Large-hf"},
            "MiDaS-Small (v2.1)":    {"t": "midas", "id": "MiDaS_small"},
            "MiDaS-Hybrid (DPT)":    {"t": "midas", "id": "DPT_Hybrid"},
            "MiDaS-Large (DPT)":     {"t": "midas", "id": "DPT_Large"},
        }
        
        cfg = MAP.get(label)
        if not cfg: raise ValueError(f"Bad model: {label}")
        self.fam = cfg["t"]

        try:
            if self.fam == "da":
                self.proc = AutoImageProcessor.from_pretrained(cfg["id"])
                self.model = AutoModelForDepthEstimation.from_pretrained(cfg["id"]).to(device)
            elif self.fam == "midas":
                self.model = torch.hub.load("intel-isl/MiDaS", cfg["id"]).to(device)
                tf_hub = torch.hub.load("intel-isl/MiDaS", "transforms")
                if cfg["id"] == "MiDaS_small":
                    self.tf = tf_hub.small_transform
                else:
                    self.tf = tf_hub.dpt_transform
            self.model.eval()
        except Exception as e:
            st.error(f"Err loading {label}: {e}")
            raise e

    def run(self, img_np):
        h, w = img_np.shape[:2]
        with torch.no_grad():
            if self.fam == "da":
                pil = Image.fromarray(img_np)
                inp = self.proc(images=pil, return_tensors="pt")
                inp = {k: v.to(self.dev) for k, v in inp.items()}
                out = self.model(**inp).predicted_depth
                out = F.interpolate(out.unsqueeze(1), size=(h, w), mode='bicubic', align_corners=False).squeeze()
            elif self.fam == "midas":
                batch = self.tf(img_np).to(self.dev)
                pred = self.model(batch)
                out = F.interpolate(pred.unsqueeze(1), size=(h, w), mode='bicubic', align_corners=False).squeeze()
        
        d = out.cpu().numpy()
        # normalize
        return (d - d.min()) / (d.max() - d.min() + 1e-8)

def plot_interactive(rgb, grasps, cog=None, rays=True):
    fig = go.Figure()
    fig.add_trace(go.Image(z=rgb))
    
    if cog:
        fig.add_trace(go.Scatter(
            x=[cog[0]], y=[cog[1]], mode='markers',
            marker=dict(color='cyan', symbol='x', size=15, line=dict(width=2, color='white')),
            name='CoG', hoverinfo='text', hovertext='Center of Gravity'
        ))

    for i, g in enumerate(grasps):
        pts = g.get_corners()
        pts = np.vstack([pts, pts[0]]) # close loop
        
        c = 'lime' if i == 0 else 'red'
        txt = (f"Rank: #{i+1}<br>Len: {g.line_length:.1f}px<br>"
               f"Score: {g.combined_quality:.2f}<br>W: {g.width:.1f}")

        fig.add_trace(go.Scatter(
            x=pts[:, 0], y=pts[:, 1], mode='lines',
            line=dict(color=c, width=3 if i==0 else 2),
            name=f'G#{i+1}', text=txt, hoverinfo='text', showlegend=False
        ))
        
        if rays and i < 3 and cog:
            # draw ray
            rad = np.radians(g.angle)
            ca, sa = np.cos(g.angle), np.sin(g.angle) # wait, angle is already rads in candidate
            ca, sa = np.cos(g.angle), np.sin(g.angle)
            hw = g.width / 2
            x1, y1 = g.x - hw*ca, g.y - hw*sa
            x2, y2 = g.x + hw*ca, g.y + hw*sa
            
            rc = 'yellow' if i==0 else 'orange'
            fig.add_trace(go.Scatter(
                x=[x1, cog[0], x2], y=[y1, cog[1], y2], mode='lines',
                line=dict(color=rc, width=2, dash='dot'), showlegend=False, hoverinfo='skip'
            ))

    fig.update_layout(
        width=700, height=500, margin=dict(l=0, r=0, b=0, t=0),
        xaxis=dict(visible=False, range=[0, rgb.shape[1]]),
        yaxis=dict(visible=False, range=[rgb.shape[0], 0], scaleanchor="x")
    )
    return fig

# ==========================================
# APP
# ==========================================

st.set_page_config(page_title="Grasp Lab", layout="wide")
st.title("Grasp Detection Lab")
st.markdown("Interactive pipeline for tuning weights and comparing depth models.")

with st.sidebar:
    st.header("Config")
    
    mod_opt = st.selectbox("Depth Model", [
        "DepthAnythingV2-Small", "DepthAnythingV2-Base", "DepthAnythingV2-Large", 
        "MiDaS-Small (v2.1)", "MiDaS-Hybrid (DPT)", "MiDaS-Large (DPT)"
    ], help="Warning: Switching unloads old model.")
    
    st.divider()
    st.subheader("Weights")
    w_edge = st.slider("Edge", 0.0, 1.0, 0.20, 0.05)
    w_depth = st.slider("Depth Grad", 0.0, 1.0, 0.40, 0.05)
    w_cog = st.slider("CoG", 0.0, 1.0, 0.40, 0.05)
    
    tot = w_edge + w_depth + w_cog
    if tot > 0:
        st.caption(f"Norm: E {w_edge/tot:.2f} | D {w_depth/tot:.2f} | C {w_cog/tot:.2f}")
    
    st.divider()
    st.subheader("Masking")
    d_pct = st.slider("Depth Percentile", 0, 100, 60, 5, help="Cutoff for background")

    st.divider()
    st.subheader("Ray Casting")
    alg = st.selectbox("Method", ["Through CoG", "Direct Line with CoG Boost"])
    
    boost_val = 0.0
    g_src = "Contour Direction (80px avg)"
    
    if alg == "Direct Line with CoG Boost":
        boost_val = st.slider("CoG Boost", 0.0, 10.0, 2.0, 0.5)
        g_src = st.selectbox("Gradient Source", 
            ["Contour Direction (80px avg)", "Depth Gradients", "Image Edges", "Radial from Center"])
            
    st.divider()
    st.subheader("Filters")
    c1, c2 = st.columns(2)
    min_len = c1.number_input("Min Len", 1, 10000, 100, 10)
    max_len = c2.number_input("Max Len", 100, 10000, 1000, 50)
    
    st.divider()
    n_out = st.number_input("Count", 1, 50, 5)
    n_mult = st.slider("Multiplier", 1, 10, 3)
    
    do_run = st.button("Run Pipeline", type="primary")

    if 'hist' in st.session_state and len(st.session_state.hist) > 0:
        st.divider()
        st.caption(f"History: {len(st.session_state.hist)}/5 runs")

# Init
if 'hist' not in st.session_state: st.session_state.hist = []
clear_memory(mod_opt)

if 'depth_wrapper' not in st.session_state:
    with st.spinner(f"Loading {mod_opt}..."):
        try:
            st.session_state.depth_wrapper = DepthWrapper(mod_opt, get_device())
            st.session_state.model_loaded = True
        except: st.stop()

# Main
st.subheader("Input")
upl = st.file_uploader("Image", type=['jpg', 'png', 'jpeg'])

if not upl: st.info("Please upload an image.")

if upl and do_run and st.session_state.get('model_loaded'):
    fbytes = np.asarray(bytearray(upl.read()), dtype=np.uint8)
    img = cv2.imdecode(fbytes, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    t_start = time.time()
    
    # Depth
    t0 = time.time()
    dmap = st.session_state.depth_wrapper.run(rgb)
    
    # Detect
    det = GraspDetector(w_edge, w_depth, w_cog)
    t1 = time.time()
    
    grasps, debug = det.process(
        rgb, dmap, n_out, d_pct, n_mult, min_len, max_len, alg, boost_val, g_src
    )
    t_detect = (time.time() - t1) * 1000
    total_ms = (time.time() - t_start) * 1000
    
    # Save result
    res = {
        "id": len(st.session_state.hist) + 1,
        "model": mod_opt,
        "weights": (w_edge, w_depth, w_cog),
        "params": f"Pct: {d_pct}, Mult: {n_mult}, Alg: {alg}",
        "grasps": grasps,
        "image": rgb,
        "depth": dmap,
        "mask": debug['mask'],
        "edges": debug['edges'],
        "grads": debug['depth_grad'],
        "time": total_ms,
        "cog": debug.get('cog'),
        "num_evaluated": debug.get('num_evaluated', 0),
        "num_valid": debug.get('num_valid', 0),
        "min_length": min_len,
        "max_length": max_len,
        "invalid_grasps": debug.get('invalid_grasps', {}),
        "has_contour": debug.get('has_contour', False),
        "top_candidates": debug.get('top_candidates', []),
        "ray_algorithm": alg
    }
    
    st.session_state.hist.insert(0, res)
    if len(st.session_state.hist) > 5: st.session_state.hist = st.session_state.hist[:5]
    
    if not grasps:
        st.error("No grasps found.")
        with st.expander("Debug Info"):
            st.write(debug) # dump raw dict

# Results View
if st.session_state.hist:
    curr = st.session_state.hist[0]
    st.subheader(f"Result #{curr['id']}")
    
    if not curr['grasps']: st.warning("No valid grasps.")
    
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Time", f"{curr['time']:.0f} ms")
    m2.metric("Evaluated", curr.get('num_evaluated', 0))
    m3.metric("Valid", curr.get('num_valid', 0))
    m4.metric("Output", len(curr['grasps']))
    m5.metric("Detect Time", f"{t_detect if 't_detect' in locals() else 0:.0f} ms") # quick fix var scope
    
    inv = curr.get('invalid_grasps', {})
    rejected = sum(inv.values())
    if rejected > 0:
        with st.expander(f"Filtered: {rejected}"):
            st.write(inv)
            
    c1, c2 = st.columns(2)
    c1.image(curr['image'], "Original", use_container_width=True)
    
    if curr['grasps']:
        fig = plot_interactive(curr['image'], curr['grasps'], curr['cog'])
        c2.plotly_chart(fig, use_container_width=True)
    else:
        # fallback viz
        over = curr['image'].copy()
        msk = curr['mask']
        over[msk > 0] = [0, 255, 0] # green tint
        if curr['cog']:
             cv2.circle(over, curr['cog'], 10, (255,0,0), -1)
        c2.image(over, "Mask + CoG (Failed)", use_container_width=True)
        
    if curr['grasps']:
        st.write("### Rankings")
        data = []
        for i, g in enumerate(curr['grasps']):
            data.append({
                "Rank": i+1, "Len": f"{g.line_length:.1f}", 
                "Score": f"{g.combined_quality:.3f}", "W": f"{g.width:.1f}", 
                "Pos": f"{g.x},{g.y}"
            })
        st.table(data)
        
    st.write("### Debug Stages")
    sc1, sc2, sc3 = st.columns(3)
    
    # 1. Candidates
    img1 = curr['image'].copy()
    cg = curr['cog']
    if cg: cv2.circle(img1, cg, 5, (255,255,0), -1)
    for i, c in enumerate(curr.get('top_candidates', [])[:20]):
        cv2.circle(img1, c['point'], 4, (0, 255, 255), 1)
    sc1.image(img1, "1. Candidates", use_container_width=True)
    
    # 2. CoG
    img2 = curr['image'].copy()
    if cg:
        cv2.circle(img2, cg, 15, (0,255,255), 2)
        cv2.drawMarker(img2, cg, (0,255,255), cv2.MARKER_CROSS, 30, 2)
    sc2.image(img2, "2. CoG Ref", use_container_width=True)
    
    # 3. Final
    img3 = curr['image'].copy()
    for g in curr['grasps']:
        pts = g.get_corners().astype(np.int32)
        cv2.polylines(img3, [pts], True, (0,255,0), 2)
    sc3.image(img3, "3. Final", use_container_width=True)
    
    with st.expander("Pipeline Internals"):
        i1, i2, i3, i4 = st.columns(4)
        i1.image(curr['depth'], "Depth", clamp=True, use_container_width=True)
        i2.image(curr['mask'], "Mask", clamp=True, use_container_width=True)
        i3.image(curr['edges'], "Edges", clamp=True, use_container_width=True)
        i4.image(curr['grads'], "Grads", clamp=True, use_container_width=True)
        
    st.divider()

# History
h1, h2 = st.columns([3,1])
h1.header("History")
if st.button("Clear"):
    st.session_state.hist = []
    st.rerun()

for item in st.session_state.hist[1:]:
    st.markdown(f"**#{item['id']}** | {item['model']} | {item['params']}")
    hc1, hc2, hc3 = st.columns([1,1,2])
    hc1.image(item['image'], "In")
    hc2.image(item['mask'], "Mask")
    hfig = plot_interactive(item['image'], item['grasps'], item['cog'])
    hfig.update_layout(height=300, width=400)
    hc3.plotly_chart(hfig, use_container_width=False)
    st.divider()