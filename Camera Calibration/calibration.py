import cv2
import numpy as np
import glob
import os
import json
import yaml

IMAGES_DIR     = "calib_images"
OUTPUT_DIR     = "calibration_output"
SQUARE_SIZE_MM = 30.0
INNER_CORNERS  = (8, 6)


def find_corners(images_dir, inner_corners, square_size):
    objp = np.zeros((inner_corners[0] * inner_corners[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:inner_corners[0], 0:inner_corners[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []
    image_size = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH +
             cv2.CALIB_CB_NORMALIZE_IMAGE +
             cv2.CALIB_CB_FAST_CHECK)

    image_paths = sorted(
        glob.glob(os.path.join(images_dir, "*.jpg")) +
        glob.glob(os.path.join(images_dir, "*.png")) +
        glob.glob(os.path.join(images_dir, "*.jpeg"))
    )

    if not image_paths:
        raise FileNotFoundError(f"No images found in '{images_dir}'")

    for fpath in image_paths:
        img  = cv2.imread(fpath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        def try_detect(src):
            r, c = cv2.findChessboardCorners(src, inner_corners, flags)
            return r, c, src

        ret, corners, gray = try_detect(gray)

        if not ret:
            ret, corners, gray = try_detect(cv2.equalizeHist(gray))

        if not ret:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            ret, corners, gray = try_detect(clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))

        if not ret:
            for sigma in (1, 2, 3):
                blurred = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), sigma)
                ret, corners, gray = try_detect(blurred)
                if ret:
                    break

        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            vis = img.copy()
            cv2.drawChessboardCorners(vis, inner_corners, corners2, ret)
            cv2.imwrite(os.path.join(OUTPUT_DIR, "corners", os.path.basename(fpath)), vis)

    return objpoints, imgpoints, image_size


def compute_per_image_error(objpoints, imgpoints, rvecs, tvecs, K, dist):
    errors = []
    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        diff = imgpoints[i] - projected
        errors.append(np.sqrt((diff ** 2).sum(axis=2).mean()))
    return errors


def calibrate(objpoints, imgpoints, image_size):
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    errors = compute_per_image_error(objpoints, imgpoints, rvecs, tvecs, K, dist)
    threshold = max(2.0 * float(np.median(errors)), 1.5)
    outliers = [i for i, e in enumerate(errors) if e > threshold]
    if outliers:
        kept_obj = [o for i, o in enumerate(objpoints) if i not in set(outliers)]
        kept_img = [o for i, o in enumerate(imgpoints) if i not in set(outliers)]
        if len(kept_obj) >= 3:
            rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                kept_obj, kept_img, image_size, None, None
            )
            objpoints[:] = kept_obj
            imgpoints[:] = kept_img

    return rms, K, dist, rvecs, tvecs


def save_results(K, dist, image_size, rms, per_image_errors, output_dir):
    np.savez(os.path.join(output_dir, "calibration.npz"),
             K=K, dist=dist, image_size=np.array(image_size), rms=np.array(rms))

    yaml_data = {
        "image_size": {"width": int(image_size[0]), "height": int(image_size[1])},
        "rms_reprojection_error_px": float(rms),
        "camera_matrix": {
            "fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            "data": K.tolist()
        },
        "distortion_coefficients": {
            "k1": float(dist[0][0]), "k2": float(dist[0][1]),
            "p1": float(dist[0][2]), "p2": float(dist[0][3]),
            "data": dist.tolist()
        }
    }
    with open(os.path.join(output_dir, "calibration.yaml"), "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    with open(os.path.join(output_dir, "per_image_errors.json"), "w") as f:
        json.dump({f"image_{i}": round(float(e), 4) for i, e in enumerate(per_image_errors)}, f, indent=2)


def main():
    os.makedirs(os.path.join(OUTPUT_DIR, "corners"), exist_ok=True)

    objpoints, imgpoints, image_size = find_corners(IMAGES_DIR, INNER_CORNERS, SQUARE_SIZE_MM)

    if len(objpoints) < 3:
        raise RuntimeError("Need at least 3 successful images.")

    rms, K, dist, rvecs, tvecs = calibrate(objpoints, imgpoints, image_size)
    per_image_errors = compute_per_image_error(objpoints, imgpoints, rvecs, tvecs, K, dist)

    save_results(K, dist, image_size, rms, per_image_errors, OUTPUT_DIR)

    print(f"\nRMS: {rms:.4f} px  ({len(objpoints)} images)")
    print(f"\nCamera Matrix:")
    print(f"  fx={K[0,0]:.2f}  fy={K[1,1]:.2f}  cx={K[0,2]:.2f}  cy={K[1,2]:.2f}")
    print(f"\nDistortion:")
    print(f"  k1={dist[0][0]:.6f}  k2={dist[0][1]:.6f}  p1={dist[0][2]:.6f}  p2={dist[0][3]:.6f}")
    print(f"\nPer-image errors:")
    for i, e in enumerate(per_image_errors):
        print(f"  img {i+1:02d}: {e:.4f} px")
    print(f"\nSaved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
