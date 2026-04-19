import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob


SCENE = "scene 2"   
SCALE = 0.5

image_paths = sorted(
    glob.glob(os.path.join(SCENE, "*.jpg")) +
    glob.glob(os.path.join(SCENE, "*.png")) +
    glob.glob(os.path.join(SCENE, "*.jpeg"))
)

from collections import Counter

calib = np.load("calibration_output/calibration.npz")
K = calib["K"]

def aspect(p):
    img = cv2.imread(p)
    h, w = img.shape[:2]
    return round(w / h, 2), img

sizes   = [aspect(p) for p in image_paths]
common_ar = Counter(ar for ar, _ in sizes).most_common(1)[0][0]
filtered  = [(p, img) for p, (ar, img) in zip(image_paths, sizes) if ar == common_ar]
skipped   = [p for p, (ar, _) in zip(image_paths, sizes) if ar != common_ar]
if skipped:
    print(f"Skipped (wrong aspect ratio): {[os.path.basename(p) for p in skipped]}")
image_paths = [p for p, _ in filtered]
raw_imgs    = [img for _, img in filtered]

images = []
for p, img in zip(image_paths, raw_imgs):
    h, w = img.shape[:2]
    img_small = cv2.resize(img, (int(w * SCALE), int(h * SCALE)), interpolation=cv2.INTER_AREA)
    images.append(img_small)


n = len(images)
print(f"\nLoaded {n} images, resized to {images[0].shape[1]}x{images[0].shape[0]}")


HARRIS_BLOCK  = 4
HARRIS_KSIZE  = 3
HARRIS_K      = 0.04
HARRIS_THRESH = 0.01

grays  = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) for img in images]
harris = [cv2.cornerHarris(g, HARRIS_BLOCK, HARRIS_KSIZE, HARRIS_K) for g in grays]

fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
if n == 1:
    axes = [axes]
fig.suptitle("Harris Corners", fontsize=14)
for i, (img, h_resp) in enumerate(zip(images, harris)):
    vis = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
    vis[h_resp > HARRIS_THRESH * h_resp.max()] = [255, 0, 0]
    axes[i].imshow(vis)
    axes[i].set_title(f"Image {i+1}")
    axes[i].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(SCENE, "harris_corners.png"), dpi=100)
plt.show()

MIN_DIST = 5

def extract_corners(h_resp, threshold_frac=HARRIS_THRESH, min_dist=MIN_DIST):
    h_norm = cv2.dilate(h_resp, None)
    mask   = (h_resp == h_norm) & (h_resp > threshold_frac * h_resp.max())
    ys, xs = np.where(mask)
    scores = h_resp[ys, xs]
    order  = np.argsort(-scores)
    kept = []
    for idx in order:
        x, y = xs[idx], ys[idx]
        if all(abs(x - kx) > min_dist or abs(y - ky) > min_dist for kx, ky in kept):
            kept.append((x, y))
    return np.array(kept, dtype=np.float32)

corners = [extract_corners(h) for h in harris]
for i, c in enumerate(corners):
    print(f"Image {i+1}: {len(c)} corners")

sift = cv2.SIFT_create()

def compute_sift_at_corners(gray_uint8, pts):
    kps = [cv2.KeyPoint(float(x), float(y), 16) for x, y in pts]
    kps, descs = sift.compute(gray_uint8, kps)
    return kps, descs

grays_uint8 = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
kp_desc = [compute_sift_at_corners(g, c) for g, c in zip(grays_uint8, corners)]

bf    = cv2.BFMatcher(cv2.NORM_L2)
RATIO = 0.75

def match_pair(desc_a, desc_b):
    raw = bf.knnMatch(desc_a, desc_b, k=2)
    good_a, good_b = [], []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, nn = pair
        if m.distance < RATIO * nn.distance:
            good_a.append(m.queryIdx)
            good_b.append(m.trainIdx)
    return good_a, good_b

pairs = [(i, i + 1) for i in range(n - 1)]
matches = {}
for a, b in pairs:
    ia, ib = match_pair(kp_desc[a][1], kp_desc[b][1])
    matches[(a, b)] = (ia, ib)
    print(f"Pair {a+1}<->{b+1}: {len(ia)} good matches")

fig, axes = plt.subplots(1, len(pairs), figsize=(7 * len(pairs), 6))
if len(pairs) == 1:
    axes = [axes]
fig.suptitle("Feature Matches", fontsize=14)
for idx, (a, b) in enumerate(pairs):
    ia, ib = matches[(a, b)]
    kpa = [kp_desc[a][0][i] for i in ia]
    kpb = [kp_desc[b][0][i] for i in ib]
    dm  = [cv2.DMatch(i, i, 0) for i in range(len(kpa))]
    vis = cv2.drawMatches(images[a], kpa, images[b], kpb, dm, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    axes[idx].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    axes[idx].set_title(f"Image {a+1} <-> Image {b+1}")
    axes[idx].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(SCENE, "matches.png"), dpi=100)
plt.show()


def get_homography(kps_a, kps_b, idx_a, idx_b):
    pts_a = np.float32([kps_a[i].pt for i in idx_a])
    pts_b = np.float32([kps_b[i].pt for i in idx_b])
    H, mask = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    print(f"  RANSAC inliers: {inliers}/{len(idx_a)}")
    return H

print("Computing pairwise homographies:")
H = {}
for a, b in pairs:
    print(f"  H{b+1}->{a+1}")
    ia, ib = matches[(a, b)]
    H[(a, b)] = get_homography(kp_desc[a][0], kp_desc[b][0], ia, ib)

ref = n // 2
H_global = [None] * n
H_global[ref] = np.eye(3)
for i in range(ref + 1, n):
    H_global[i] = H_global[i - 1] @ H[(i - 1, i)]
for i in range(ref - 1, -1, -1):
    H_global[i] = H_global[i + 1] @ np.linalg.inv(H[(i, i + 1)])
print(f"Reference frame: image {ref + 1} (center)")

def warp_corners(img, Hg):
    h, w = img.shape[:2]
    c = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(c, Hg).reshape(-1, 2)

all_corners = np.vstack([warp_corners(img, Hg) for img, Hg in zip(images, H_global)])
x_min, y_min = all_corners.min(axis=0)
x_max, y_max = all_corners.max(axis=0)
x_min, y_min = int(np.floor(x_min)), int(np.floor(y_min))
x_max, y_max = int(np.ceil(x_max)),  int(np.ceil(y_max))
canvas_w, canvas_h = x_max - x_min, y_max - y_min
print(f"Canvas size: {canvas_w} x {canvas_h}")

T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
H_shifted = [T @ Hg for Hg in H_global]

canvas_sum   = np.zeros((canvas_h, canvas_w, 3), dtype=np.float64)
canvas_count = np.zeros((canvas_h, canvas_w),    dtype=np.float64)
for img, Hg in zip(images, H_shifted):
    warped = cv2.warpPerspective(img.astype(np.float64), Hg, (canvas_w, canvas_h))
    mask_w = cv2.warpPerspective(np.ones(img.shape[:2], dtype=np.float64), Hg, (canvas_w, canvas_h))
    canvas_sum   += warped
    canvas_count += mask_w

with np.errstate(invalid='ignore', divide='ignore'):
    mosaic = np.where(canvas_count[..., None] > 0,
                      canvas_sum / canvas_count[..., None], 0).astype(np.uint8)

plt.figure(figsize=(20, 8))
plt.imshow(cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))
plt.title("Final Panorama", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(SCENE, "mosaic.png"), dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved to {SCENE}/")
