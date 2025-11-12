
import os
import json
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Optional
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


ENCODER = "vits"  # 'vits' | 'vitb' | 'vitl' | 'vitg'
MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}
WEIGHTS_PATH = f"depth_anything_v2_{ENCODER}.pth"

IMAGES_DIR = Path(r"C:\Users\samst\OneDrive\Desktop\ml_final_project\shirtv1\roe1\synthetic\images")
CAMERA_JSON = Path(r"C:\Users\samst\OneDrive\Desktop\ml_final_project\shirtv1\camera.json")
POSES_JSON  = Path(r"C:\Users\samst\OneDrive\Desktop\ml_final_project\shirtv1\roe1\roe1.json")
OUT_DIR = Path("./outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SINGLE_VIEW_FILE = "img000033.jpg"


#FROM CAMERA JSON
PREVIEW_FX = 3020.0062181662161
PREVIEW_CX = 960
PREVIEW_CY = 600

def load_model(encoder: str, weights_path: Path, device: str) -> DepthAnythingV2:
    model = DepthAnythingV2(**MODEL_CONFIGS[encoder])
    sd = torch.load(str(weights_path), map_location='cpu')
    model.load_state_dict(sd)
    model = model.to(device).eval()
    return model


def save_depth_visualizations(depth: np.ndarray, stem: str) -> None:
    d = depth.astype(np.float32)
    dmin, dmax = np.nanmin(d), np.nanmax(d)
    if not np.isfinite(dmin) or not np.isfinite(dmax) or dmax <= dmin:
        dmin, dmax = 0.0, 1.0
    d_norm = np.clip((d - dmin) / (dmax - dmin + 1e-12), 0, 1)

    gray_u8 = (d_norm * 255).astype(np.uint8)
    cv2.imwrite(str(OUT_DIR / f"{stem}_depth_gray.png"), gray_u8)

    plt.figure(figsize=(10, 7))
    plt.imshow(d, cmap='plasma')
    plt.colorbar(label='Depth (relative)')
    plt.title('Depth Map')
    plt.axis('off')
    plt.savefig(OUT_DIR / f"{stem}_depth_color.png", bbox_inches='tight', dpi=200)
    plt.close()


def isolate_center_object(depth_map: np.ndarray,
                          image: np.ndarray,
                          pad_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    d = depth_map.astype(np.float32).copy()
    h, w = d.shape
    d_smooth = cv2.bilateralFilter(d, 9, 0.2, 10)

    center_depth = d_smooth[h // 2, w // 2]

    diff = np.abs(d_smooth - center_depth)
    diff_u8 = (255.0 * diff / (diff.max() + 1e-8)).astype(np.uint8)
    _, mask = cv2.threshold(diff_u8, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)


    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels = cv2.connectedComponents(mask)
    if num_labels > 1:
        label_counts = np.bincount(labels.flat)[1:]  # Exclude background
        best_label = np.argmax(label_counts) + 1
        mask = (labels == best_label).astype(np.uint8) * 255

    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return d, image, (0, 0, w - 1, h - 1)

    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    pad_y = int((y2 - y1 + 1) * pad_ratio)
    pad_x = int((x2 - x1 + 1) * pad_ratio)
    y1 = max(0, y1 - pad_y)
    y2 = min(h - 1, y2 + pad_y)
    x1 = max(0, x1 - pad_x)
    x2 = min(w - 1, x2 + pad_x)

    mask_c = mask[y1:y2 + 1, x1:x2 + 1]
    depth_c = d[y1:y2 + 1, x1:x2 + 1]
    img_c = image[y1:y2 + 1, x1:x2 + 1]

    depth_c_masked = depth_c.copy()
    depth_c_masked[mask_c == 0] = np.nan

    return depth_c_masked, img_c, (x1, y1, x2, y2)


def depth_to_point_cloud(depth_map: np.ndarray,
                         image_bgr: np.ndarray,
                         fx: float,
                         cx: float,
                         cy: float,
                         crop_offset_x: int = 0,
                         crop_offset_y: int = 0) -> Tuple[np.ndarray, np.ndarray]:

    h, w = depth_map.shape
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_map.astype(np.float32)

    cx_adjusted = cx - crop_offset_x
    cy_adjusted = cy - crop_offset_y

    x_cam = (xs - cx_adjusted) * z / float(fx)
    y_cam = (ys - cy_adjusted) * z / float(fx)  

    pts = np.stack([x_cam, y_cam, z], axis=-1).reshape(-1, 3)

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1) & (z.reshape(-1) > 0)

    return pts[valid], rgb[valid].astype(np.uint8)


def save_ply(points: np.ndarray, colors: np.ndarray, path: Path) -> None:
    """Save ASCII PLY: (x,y,z,uint8 r,g,b)."""
    path = Path(path)
    with path.open('w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(points, colors):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")



def load_camera_and_pose_data(camera_json: Path, poses_json: Path):
    with camera_json.open('r') as f:
        camera_data = json.load(f)
    with poses_json.open('r') as f:
        poses_data = json.load(f)
    return camera_data, poses_data

def depth_to_world_with_pose(depth_map: np.ndarray,
                             image_bgr: np.ndarray,
                             camera_intrinsics: dict,
                             T_world_from_cam: np.ndarray,
                             crop_offset_x: int = 0,
                             crop_offset_y: int = 0) -> Tuple[np.ndarray, np.ndarray]:
 
    K = np.array(camera_intrinsics['cameraMatrix'], dtype=np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    h, w = depth_map.shape
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_map.astype(np.float32)

    cx_adjusted = cx - crop_offset_x
    cy_adjusted = cy - crop_offset_y

    x_cam = (xs - cx_adjusted) * z / float(fx)
    y_cam = (ys - cy_adjusted) * z / float(fy)
    pts_cam = np.stack([x_cam, y_cam, z], axis=-1).reshape(-1, 3)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).reshape(-1, 3)

    valid = np.isfinite(pts_cam).all(axis=1) & (z.reshape(-1) > 0)
    pts_cam = pts_cam[valid]
    rgb = rgb[valid].astype(np.uint8)
    #return pts_cam, rgb

    # to world
    pts_cam_h = np.hstack([pts_cam, np.ones((pts_cam.shape[0], 1), dtype=np.float64)])
    pts_w_h = (T_world_from_cam @ pts_cam_h.T).T
    pts_w = pts_w_h[:, :3]
    return pts_w, rgb



def run_single_view(model: DepthAnythingV2, example_file: str) -> None:
    """Single-view: depth inference, isolation, point cloud"""
    img_path = IMAGES_DIR / example_file
    bgr = cv2.imread(str(img_path))

    with torch.no_grad():
        depth = model.infer_image(bgr)  

    save_depth_visualizations(depth, stem=img_path.stem)

    depth_masked, bgr_crop, bbox = isolate_center_object(depth, bgr)

    dm = depth_masked.copy()
    dmin, dmax = np.nanmin(dm), np.nanmax(dm)
    if np.isfinite(dmin) and np.isfinite(dmax) and dmax > dmin:
        dm_norm = ((dm - dmin) / (dmax - dmin + 1e-12) * 255).astype(np.uint8)
    else:
        dm_norm = np.zeros_like(dm, dtype=np.uint8)
    cv2.imwrite(str(OUT_DIR / f"{img_path.stem}_masked_depth.png"), dm_norm)

    x1, y1, x2, y2 = bbox
    pts, cols = depth_to_point_cloud(depth_masked, bgr_crop, fx=PREVIEW_FX, cx=PREVIEW_CX, cy=PREVIEW_CY, 
                                     crop_offset_x=x1, crop_offset_y=y1)


    ply_path = OUT_DIR / f"{img_path.stem}_point_cloud.ply"
    save_ply(pts, cols, ply_path)
    print(f"[Single-View] Saved point cloud: {ply_path}  (N={len(pts)})")




# ------------------- MAIN -------------------
def main():
    model = load_model(ENCODER, Path(WEIGHTS_PATH), DEVICE)
    run_single_view(model, SINGLE_VIEW_FILE)

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
