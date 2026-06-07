"""
ML-SLAM Room Mapper V9
======================
LightGlue (Apache-2.0, open-source) + DISK feature extractor
Pretrained weights automatically download on first run — kuch manually nahi karna!

SETUP (ek baar sirf):
----------------------
  pip install opencv-python open3d numpy torch torchvision
  pip install git+https://github.com/cvg/LightGlue.git

RUN:
----
  python ml_slam_v9.py

First run pe ~100MB weights automatically download honge (Hugging Face se).
Dobara run pe cache se load hoga — fast.
"""

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# LIGHTGLUE IMPORT — agar install nahi toh helpful error deta hai
# ─────────────────────────────────────────────────────────────────────────────
try:
    from lightglue import LightGlue, DISK, SuperPoint as LG_SuperPoint
    from lightglue.utils import rbd   # remove batch dimension helper
    LIGHTGLUE_AVAILABLE = True
    print("✅ LightGlue import successful")
except ImportError:
    LIGHTGLUE_AVAILABLE = False
    print("❌ LightGlue nahi mila!")
    print("\n   Install karo:")
    print("   pip install git+https://github.com/cvg/LightGlue.git")
    print("\n   Ya DISK ke liye:")
    print("   pip install git+https://github.com/cvg/LightGlue.git")
    print("\nFallback: ORB + FLANN use ho raha hai...\n")

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {DEVICE}  (GPU mila toh fast, CPU pe thoda slow — dono chalenge)")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD LIGHTGLUE MODELS
# Weights pehli baar automatically download honge (~100MB)
# Cache: C:\Users\<user>\.cache\torch\hub\checkpoints\
# ─────────────────────────────────────────────────────────────────────────────
extractor = None
matcher   = None

if LIGHTGLUE_AVAILABLE:
    print("\n🤖 ML models load ho rahe hain...")
    print("   (Pehli baar chalane pe ~100MB weights download honge — thoda wait karo)")

    try:
        # DISK: rotation/scale invariant, ORB se bahut better indoor scenes mein
        # SuperPoint bhi available hai but license restricted — DISK better choice
        extractor = DISK(max_num_keypoints=1024).eval().to(DEVICE)
        matcher   = LightGlue(features="disk").eval().to(DEVICE)
        print("✅ DISK + LightGlue ready!")
        ML_BACKEND = "DISK+LightGlue"

    except Exception as e:
        print(f"⚠️  DISK load fail ({e})")
        try:
            # Fallback to SuperPoint within LightGlue
            extractor = LG_SuperPoint(max_num_keypoints=1024).eval().to(DEVICE)
            matcher   = LightGlue(features="superpoint").eval().to(DEVICE)
            print("✅ SuperPoint + LightGlue ready!")
            ML_BACKEND = "SuperPoint+LightGlue"
        except Exception as e2:
            print(f"⚠️  SuperPoint bhi fail ({e2}) — ORB fallback use hoga")
            extractor = None
            matcher   = None
            ML_BACKEND = "ORB+FLANN (fallback)"
else:
    ML_BACKEND = "ORB+FLANN (fallback)"

print(f"\n🎯 Active backend: {ML_BACKEND}\n")

# ─────────────────────────────────────────────────────────────────────────────
# ORB FALLBACK (agar LightGlue available nahi)
# ─────────────────────────────────────────────────────────────────────────────
orb_fb   = cv2.ORB_create(nfeatures=3000, fastThreshold=10)
flann_fb = cv2.FlannBasedMatcher(
    dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=2),
    dict(checks=100)
)

# ─────────────────────────────────────────────────────────────────────────────
# CAMERA INTRINSICS
# ─────────────────────────────────────────────────────────────────────────────
fx, fy = 600.0, 600.0
cx, cy = 320.0, 240.0
K = np.array([[fx,  0, cx],
              [ 0, fy, cy],
              [ 0,  0,  1]], dtype=np.float64)
IMG_W, IMG_H = 640, 480

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────────────────────
global_pcd    = o3d.geometry.PointCloud()
global_pose   = np.eye(4)
is_first_scan = True
scan_count    = 0
pose_history  = []
prev_data     = None   # dict with frame data


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: CLAHE preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(frame):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return gray, clahe.apply(gray)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: LightGlue feature extraction
# Input: BGR frame
# Output: feats dict (LightGlue format), keypoints numpy (N,2) for display
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_features_lg(frame_bgr):
    """DISK/SuperPoint via LightGlue extractor."""
    # LightGlue expects RGB float32 tensor (3, H, W) in [0,1]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t   = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)  # (1,3,H,W)
    feats = extractor.extract(t)   # returns dict with keypoints, descriptors, scores
    return feats


def feats_to_numpy_kpts(feats):
    """For display only — extract (N,2) numpy keypoints."""
    kpts = feats["keypoints"][0].cpu().numpy()  # (N, 2)
    return kpts


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: LightGlue matching
# Returns: (src_pts, dst_pts, n_matches) — numpy arrays shape (M,2)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def match_lg(feats0, feats1):
    matches01 = matcher({"image0": feats0, "image1": feats1})

    # rbd removes batch dimension → clean dicts
    f0 = rbd(feats0)
    f1 = rbd(feats1)
    m  = rbd(matches01)

    kpts0 = f0["keypoints"].cpu().numpy()       # (N, 2)
    kpts1 = f1["keypoints"].cpu().numpy()       # (M, 2)
    matches = m["matches"].cpu().numpy()        # (K, 2) — pairs of indices

    if len(matches) < 10:
        return None, None, len(matches)

    src = kpts0[matches[:, 0]]   # (K, 2)
    dst = kpts1[matches[:, 1]]   # (K, 2)
    return src, dst, len(matches)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: ORB fallback matching
# ─────────────────────────────────────────────────────────────────────────────
def match_orb_fallback(gray_enh0, gray_enh1):
    kp0, des0 = orb_fb.detectAndCompute(gray_enh0, None)
    kp1, des1 = orb_fb.detectAndCompute(gray_enh1, None)

    if des0 is None or des1 is None or len(kp0) < 10 or len(kp1) < 10:
        return None, None, 0

    raw  = flann_fb.knnMatch(des0, des1, k=2)
    good = [m for m, n in raw if m.distance < 0.7 * n.distance]

    if len(good) < 15:
        return None, None, len(good)

    src = np.float32([kp0[m.queryIdx].pt for m in good])
    dst = np.float32([kp1[m.trainIdx].pt for m in good])
    return src, dst, len(good)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Pose from matched point pairs
# ─────────────────────────────────────────────────────────────────────────────
def estimate_pose(src_pts, dst_pts, scale=0.25):
    if src_pts is None or len(src_pts) < 10:
        return False, np.eye(4), 0

    src = src_pts.reshape(-1, 1, 2).astype(np.float32)
    dst = dst_pts.reshape(-1, 1, 2).astype(np.float32)

    E, mask = cv2.findEssentialMat(
        src, dst, K, method=cv2.RANSAC, prob=0.999, threshold=0.5
    )
    if E is None:
        return False, np.eye(4), 0

    inliers = int(mask.sum())
    if inliers < 10:
        return False, np.eye(4), inliers

    _, R, t, _ = cv2.recoverPose(E, src, dst, K, mask=mask)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t.squeeze() * scale
    return True, T, inliers


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Monocular pseudo-depth
# ─────────────────────────────────────────────────────────────────────────────
def estimate_depth(gray):
    lap   = cv2.Laplacian(gray, cv2.CV_64F)
    sharp = cv2.normalize(np.abs(lap), None, 0.0, 1.0, cv2.NORM_MINMAX).astype(np.float32)
    return np.clip(3.0 - sharp * 2.5, 0.3, 3.5)


def frame_to_pcd(frame, gray):
    h, w = gray.shape
    step = 3
    us, vs = np.meshgrid(np.arange(0, w, step), np.arange(0, h, step))
    u_f, v_f = us.flatten(), vs.flatten()
    z = estimate_depth(gray)[v_f, u_f]
    mask = z > 0.3
    u_f, v_f, z = u_f[mask], v_f[mask], z[mask]
    x = (u_f - cx) * z / fx
    y = (v_f - cy) * z / fy
    pts  = np.stack([x, -y, z], axis=-1)
    cols = frame[v_f, u_f][:, ::-1] / 255.0
    pcd  = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd


def clean_pcd(pcd, voxel=0.02):
    if len(pcd.points) < 100:
        return pcd
    pcd = pcd.voxel_down_sample(voxel)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30)
    )
    pcd.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))
    return pcd


def icp_refine(src_pcd, tgt_pcd, init_T, max_dist=0.25):
    if len(src_pcd.points) < 50 or len(tgt_pcd.points) < 50:
        return init_T, 0.0
    result = o3d.pipelines.registration.registration_icp(
        src_pcd, tgt_pcd, max_dist, init_T,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(1e-6, 1e-6, 120)
    )
    return result.transformation, result.fitness


# ─────────────────────────────────────────────────────────────────────────────
# CAMERA SETUP
# ─────────────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  IMG_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_H)
cap.set(cv2.CAP_PROP_FPS, 30)

print("=" * 60)
print(f"  🤖 ML-SLAM V9  |  Backend: {ML_BACKEND}")
print("=" * 60)
print("  's'  → Scan lo (map mein add)")
print("  'v'  → Map dekho")
print("  'r'  → Reset")
print("  'q'  → Save karke quit")
print("=" * 60)
print("\n  💡 TIPS:")
print("   • Textured walls saamne rakho")
print("   • 10-15 degree hi ghoomna ek scan se doosre tak")
print("   • 50%+ overlap har scan mein\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray, gray_enh = preprocess(frame)

    # Live feature detection for display
    display = frame.copy()
    if extractor is not None:
        try:
            feats_live = extract_features_lg(frame)
            kpts_live  = feats_to_numpy_kpts(feats_live)
            for pt in kpts_live[:400]:
                cv2.circle(display, (int(pt[0]), int(pt[1])), 2, (0, 220, 0), -1)
            kpt_count = len(kpts_live)
        except Exception:
            kpt_count = 0
            feats_live = None
    else:
        kp_live, _ = orb_fb.detectAndCompute(gray_enh, None)
        for k in kp_live[:400]:
            cv2.circle(display, (int(k.pt[0]), int(k.pt[1])), 2, (0, 220, 0), -1)
        kpt_count = len(kp_live)
        feats_live = None

    status = "FIRST SCAN → press 's'" if is_first_scan else f"Scans: {scan_count}"
    cv2.putText(display, f"{status}  [{ML_BACKEND}]",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)
    cv2.putText(display, f"Keypoints: {kpt_count}",
                (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

    cv2.imshow("ML-SLAM Live", display)
    key = cv2.waitKey(1) & 0xFF

    # ── SCAN ─────────────────────────────────────────────────────────────────
    if key == ord('s'):
        scan_count += 1
        print(f"\n[SCAN #{scan_count}] ----------------------------------------")

        new_pcd = frame_to_pcd(frame, gray)
        new_pcd = clean_pcd(new_pcd)

        if len(new_pcd.points) < 100:
            print("   ❌ Points bahut kam — achhi lighting mein try karo!")
            scan_count -= 1
            continue

        # ── FIRST SCAN ───────────────────────────────────────────────────────
        if is_first_scan:
            global_pcd    = new_pcd
            is_first_scan = False
            pose_history.append(global_pose[:3, 3].copy())

            # Store current frame for next scan
            if extractor is not None:
                prev_feats = extract_features_lg(frame) if feats_live is None else feats_live
            else:
                prev_feats = None
            prev_data = {"feats": prev_feats, "gray_enh": gray_enh, "frame": frame.copy()}
            print(f"   ✅ Base scan ready! ({len(global_pcd.points)} pts)")

        # ── SUBSEQUENT SCANS ─────────────────────────────────────────────────
        else:
            # Matching
            if extractor is not None and prev_data["feats"] is not None:
                curr_feats = extract_features_lg(frame) if feats_live is None else feats_live
                src_pts, dst_pts, n_matches = match_lg(prev_data["feats"], curr_feats)
                used_ml = True
                tag = f"🤖 {ML_BACKEND}"
            else:
                src_pts, dst_pts, n_matches = match_orb_fallback(
                    prev_data["gray_enh"], gray_enh
                )
                used_ml = False
                tag = "🔧 ORB fallback"
                curr_feats = None

            print(f"   {tag}: {n_matches} matches")

            # If ML gave too few matches, try ORB fallback
            if n_matches < 10 and used_ml:
                print(f"   ↩️  ML matches kam — ORB fallback try kar raha hai...")
                src_pts, dst_pts, n_matches = match_orb_fallback(
                    prev_data["gray_enh"], gray_enh
                )
                print(f"   🔧 ORB: {n_matches} matches")

            # Pose estimation
            vo_ok, T_delta, n_inliers = estimate_pose(src_pts, dst_pts)

            if vo_ok:
                init_T = global_pose @ T_delta
                print(f"   📍 Pose: {n_inliers} RANSAC inliers")
            else:
                T_fb = np.eye(4); T_fb[2, 3] = 0.15
                init_T = global_pose @ T_fb
                print(f"   ⚠️  Pose fail — 15cm forward fallback")

            # ICP refinement
            refined_T, fitness = icp_refine(
                o3d.geometry.PointCloud(new_pcd), global_pcd, init_T
            )
            print(f"   🔧 ICP fitness: {fitness:.4f}", end="")

            if fitness > 0.05:
                global_pose = refined_T
                pose_history.append(global_pose[:3, 3].copy())
                new_pcd.transform(refined_T)
                global_pcd += new_pcd
                global_pcd = clean_pcd(global_pcd, voxel=0.02)
                print(f"   ✅ Added! Total: {len(global_pcd.points)} pts")
            else:
                print(f"\n   ❌ Poor alignment ({fitness:.4f}) — camera ko overlap rakho!")
                scan_count -= 1
                # Still update prev_data so next attempt uses latest frame
                prev_data = {
                    "feats": curr_feats if extractor else None,
                    "gray_enh": gray_enh,
                    "frame": frame.copy()
                }
                continue

        # Update prev_data
        if extractor is not None:
            curr_feats_store = extract_features_lg(frame)
        else:
            curr_feats_store = None
        prev_data = {
            "feats": curr_feats_store,
            "gray_enh": gray_enh,
            "frame": frame.copy()
        }

        # Visualize
        viz = [global_pcd]
        for i, pos in enumerate(pose_history):
            s = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
            s.translate(pos)
            s.paint_uniform_color([1.0, 0.0, 0.0] if i == len(pose_history)-1 else [0.0, 1.0, 0.3])
            viz.append(s)
        viz.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2))

        print(f"   🖼️  Close visualization window → next scan")
        o3d.visualization.draw_geometries(
            viz,
            window_name=f"ML-SLAM V9 — {scan_count} scans | {len(global_pcd.points)} pts",
            width=1100, height=700
        )

    # ── VIEW MAP ─────────────────────────────────────────────────────────────
    elif key == ord('v'):
        if not is_first_scan:
            o3d.visualization.draw_geometries(
                [global_pcd],
                window_name=f"Map ({scan_count} scans)",
                width=1100, height=700
            )
        else:
            print("\n⚠️  Pehle 's' se scans lo!")

    # ── RESET ────────────────────────────────────────────────────────────────
    elif key == ord('r'):
        global_pcd  = o3d.geometry.PointCloud()
        global_pose = np.eye(4)
        is_first_scan = True
        scan_count  = 0
        pose_history.clear()
        prev_data = None
        print("\n[RESET] ✅ Fresh start!")

    # ── QUIT ─────────────────────────────────────────────────────────────────
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ─────────────────────────────────────────────────────────────────────────────
# SAVE FINAL MAP
# ─────────────────────────────────────────────────────────────────────────────
if not is_first_scan and len(global_pcd.points) > 0:
    out = "ml_slam_room_map_v9.ply"
    o3d.io.write_point_cloud(out, global_pcd)
    print(f"\n🎉 Map saved: '{out}'")
    print(f"   Points : {len(global_pcd.points)}")
    print(f"   Scans  : {scan_count}")
    print(f"   Backend: {ML_BACKEND}")
    o3d.visualization.draw_geometries(
        [global_pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)],
        window_name="Final ML-SLAM Map",
        width=1200, height=800
    )
else:
    print("\n⚠️  Koi scan nahi — nothing to save.")