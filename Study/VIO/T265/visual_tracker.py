"""
Stereo Visual Odometry front-end for the T265 (Kannala-Brandt fisheye).

Pipeline per stereo pair:
  1. Undistort both fisheye images → pinhole images (cv2.fisheye)
  2. [Temporal]  LK-track prev-left features → curr-left
  3. [Pose]      solvePnPRansac on 3D↔2D correspondences → metric R, t
  4. [Stereo]    LK-match curr-left features → curr-right
  5. [Depth]     Triangulate → metric 3-D landmark positions
  6. Save curr-left + 3-D points for next frame

Advantages over monocular:
  - Metric scale from stereo baseline (no IMU scale estimation)
  - PnP instead of Essential-matrix decomposition (more accurate)
  - Richer 3-D map for EKF update
"""

import cv2
import numpy as np


def _intr_to_K(intr) -> np.ndarray:
    """pyrealsense2 intrinsics → 3×3 camera matrix."""
    return np.array([[intr.fx, 0, intr.ppx],
                     [0, intr.fy, intr.ppy],
                     [0,       0,        1]], dtype=np.float64)


def _intr_to_D(intr) -> np.ndarray:
    """pyrealsense2 intrinsics → distortion coeffs [k1,k2,k3,k4]."""
    return np.array(intr.coeffs[:4], dtype=np.float64)


class StereoVisualTracker:
    """
    Stereo VO tracker for T265 fisheye cameras.

    Args:
        calib: dict with keys intr1, intr2, R_stereo, t_stereo
               (from T265Reader.get_stereo_calibration())
        cfg:   optional tuning dict
    """

    # Fallback approximate intrinsics for T265 (used if no hardware calib)
    _DEFAULT_K = np.array([[287., 0., 424.],
                            [0., 287., 404.],
                            [0.,   0.,   1.]], dtype=np.float64)
    _DEFAULT_D = np.zeros(4)
    _DEFAULT_BASELINE = 0.064   # metres

    def __init__(self, calib=None, cfg=None):
        cfg = cfg or {}

        if calib is not None:
            self.K1 = _intr_to_K(calib["intr1"])
            self.D1 = _intr_to_D(calib["intr1"])
            self.K2 = _intr_to_K(calib["intr2"])
            self.D2 = _intr_to_D(calib["intr2"])
            self.R_stereo = calib["R_stereo"].astype(np.float64)
            self.t_stereo = calib["t_stereo"].astype(np.float64)
        else:
            self.K1 = self.K2 = self._DEFAULT_K.copy()
            self.D1 = self.D2 = self._DEFAULT_D.copy()
            self.R_stereo = np.eye(3)
            self.t_stereo = np.array([-self._DEFAULT_BASELINE, 0., 0.])

        self.baseline = np.linalg.norm(self.t_stereo)

        # Output (pinhole) camera matrix after undistortion.
        # Using K1 directly keeps the same focal length as the original fisheye,
        # so objects appear at natural size.  estimateNewCameraMatrix with low
        # balance produces K_new.fx << K1.fx → extreme zoom-out / "spread-out"
        # appearance.  Beyond the ±56° half-angle the image has black borders,
        # which is acceptable and actually helps understand the valid FOV.
        self.K_new = self.K1.copy()

        # Projection matrices for triangulation
        # Left:  P1 = K_new [I | 0]
        # Right: P2 = K_new [R | t]
        self.P1 = self.K_new @ np.hstack([np.eye(3), np.zeros((3, 1))])
        self.P2 = self.K_new @ np.hstack([self.R_stereo,
                                           self.t_stereo.reshape(3, 1)])

        # LK params
        self._lk = dict(
            winSize  = (cfg.get("lk_win", 21),) * 2,
            maxLevel = cfg.get("lk_levels", 3),
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        cfg.get("lk_iters", 30), 0.01),
        )

        # Feature detection
        self._max_corners  = cfg.get("max_corners", 250)
        self._quality      = cfg.get("quality",     0.01)
        self._min_dist     = cfg.get("min_dist",    12)
        self._min_inliers  = cfg.get("min_inliers", 12)

        # Depth filter [m].
        # Stereo depth error ≈ depth² / (f × baseline).  At 2 m with f=287, b=0.064:
        # σ_depth ≈ 4 / (287×0.064) ≈ 22 cm/pixel → keep features close.
        self._z_min = cfg.get("z_min", 0.15)
        self._z_max = cfg.get("z_max", 2.0)

        # State between frames
        self._prev_und: np.ndarray | None = None
        self._prev_pts2d: np.ndarray | None = None
        self._prev_pts3d: np.ndarray | None = None

        # Debug info populated each process() call (read by visualizer)
        self.last_debug: dict = {}

    # ── public API ─────────────────────────────────────────────────────────

    def process(self, img_left: np.ndarray, img_right: np.ndarray):
        """
        Feed a new stereo pair.
        Returns (R, t_metric, n_inliers) or None if not enough data yet.
        t_metric is in the LEFT camera frame of the PREVIOUS frame.
        """
        und_l = self._undistort(img_left,  self.K1, self.D1)
        und_r = self._undistort(img_right, self.K2, self.D2)

        dbg = {'und_l': und_l, 'und_r': und_r,
               'tracked_prev': None, 'tracked_curr': None, 'tracked_inliers': None,
               'stereo_l': None, 'stereo_r': None, 'depths': None}

        result = None

        # ── Step 1: temporal tracking + PnP ──────────────────────────────
        if (self._prev_und is not None
                and self._prev_pts2d is not None
                and self._prev_pts3d is not None
                and len(self._prev_pts2d) >= self._min_inliers):

            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_und, und_l,
                self._prev_pts2d.reshape(-1, 1, 2), None, **self._lk
            )
            ok = (status.ravel() == 1)
            if ok.sum() >= self._min_inliers:
                pts3d = self._prev_pts3d[ok]
                pts2d = curr_pts[ok].reshape(-1, 2)

                dbg['tracked_prev'] = self._prev_pts2d[ok].copy()
                dbg['tracked_curr'] = pts2d.copy()

                result = self._pnp(pts3d, pts2d)
                if result is not None:
                    R_res, t_res, n_inl, inl_idx = result
                    inlier_mask = np.zeros(len(pts2d), dtype=bool)
                    inlier_mask[inl_idx] = True
                    dbg['tracked_inliers'] = inlier_mask
                    # Transform inlier 3D pts from prev-cam frame → curr-cam frame.
                    # solvePnP gives: p_curr = R @ p_prev + t
                    # All stored pts3d must be in the same (current) frame so that
                    # next frame's PnP sees a consistent coordinate system.
                    inl_3d_prev = pts3d[inl_idx]
                    inl_3d_curr = (R_res @ inl_3d_prev.T
                                   + t_res.reshape(3, 1)).T.astype(np.float32)
                    self._prev_pts2d = pts2d[inl_idx].reshape(-1, 2)
                    self._prev_pts3d = inl_3d_curr
                    result = (R_res, t_res, n_inl)

        # ── Step 2: detect features in left, match to right ──────────────
        new_pts_l = self._detect(und_l)
        if new_pts_l is not None and len(new_pts_l) >= self._min_inliers:
            pts3d_new, pts2d_new, pts2d_r_new = self._stereo_triangulate(
                und_l, und_r, new_pts_l
            )
            if pts3d_new is not None and len(pts3d_new) >= self._min_inliers:
                dbg['stereo_l']  = pts2d_new.copy()
                dbg['stereo_r']  = pts2d_r_new.copy()
                dbg['depths']    = pts3d_new[:, 2].copy()
                if (self._prev_pts2d is not None
                        and len(self._prev_pts2d) > 0
                        and self._prev_pts3d is not None):
                    self._prev_pts2d = np.vstack([self._prev_pts2d, pts2d_new])
                    self._prev_pts3d = np.vstack([self._prev_pts3d, pts3d_new])
                else:
                    self._prev_pts2d = pts2d_new
                    self._prev_pts3d = pts3d_new
                # Cap total points so LK/PnP stay O(max_corners).
                # PnP inliers come first → they are preserved by the slice.
                if len(self._prev_pts2d) > self._max_corners:
                    self._prev_pts2d = self._prev_pts2d[:self._max_corners]
                    self._prev_pts3d = self._prev_pts3d[:self._max_corners]

        self._prev_und = und_l
        self.last_debug = dbg
        return result

    def reset(self):
        self._prev_und   = None
        self._prev_pts2d = None
        self._prev_pts3d = None

    # ── internal helpers ───────────────────────────────────────────────────

    def _undistort(self, img: np.ndarray, K, D) -> np.ndarray:
        """Kannala-Brandt fisheye → pinhole undistortion."""
        h, w = img.shape[:2]
        return cv2.fisheye.undistortImage(img, K, D, None, self.K_new,
                                          (w, h))

    def _detect(self, gray: np.ndarray) -> np.ndarray | None:
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners   = self._max_corners,
            qualityLevel = self._quality,
            minDistance  = self._min_dist,
            useHarrisDetector=False,
        )

    def _stereo_triangulate(self, und_l, und_r, pts_l):
        """
        Match pts_l to und_r via LK, filter by epipolar + depth,
        triangulate → (pts3d, pts2d_left, pts2d_right).
        """
        pts_l = pts_l.reshape(-1, 1, 2).astype(np.float32)
        pts_r, status, _ = cv2.calcOpticalFlowPyrLK(
            und_l, und_r, pts_l, None, **self._lk
        )
        ok = (status.ravel() == 1)

        if ok.sum() < self._min_inliers:
            return None, None, None

        pl = pts_l[ok].reshape(-1, 2)
        pr = pts_r[ok].reshape(-1, 2)

        # T265 cameras are nearly parallel → vertical disparity should be small
        dy = np.abs(pl[:, 1] - pr[:, 1])
        epi_ok = dy < 3.0
        pl, pr = pl[epi_ok], pr[epi_ok]

        if len(pl) < self._min_inliers:
            return None, None, None

        pts4d = cv2.triangulatePoints(
            self.P1, self.P2,
            pl.T.astype(np.float64),
            pr.T.astype(np.float64),
        )
        pts3d = (pts4d[:3] / (pts4d[3] + 1e-9)).T

        valid = (pts3d[:, 2] > self._z_min) & (pts3d[:, 2] < self._z_max)
        if valid.sum() < self._min_inliers:
            return None, None, None

        return (pts3d[valid].astype(np.float32),
                pl[valid].astype(np.float32),
                pr[valid].astype(np.float32))

    def _pnp(self, pts3d, pts2d):
        """
        solvePnPRansac → (R, t, n_inliers, inlier_indices) or None.
        pts3d: (N,3) float32, pts2d: (N,2) float32
        """
        if len(pts3d) < self._min_inliers:
            return None
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts3d.astype(np.float64),
                pts2d.astype(np.float64),
                self.K_new, None,
                iterationsCount = 100,
                reprojectionError = 2.0,
                confidence = 0.999,
                flags = cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            return None

        if not ok or inliers is None or len(inliers) < self._min_inliers:
            return None

        R, _ = cv2.Rodrigues(rvec)
        t    = tvec.ravel()
        inl  = inliers.ravel()

        # Reject physically implausible translations (> 30 cm per frame)
        if np.linalg.norm(t) > 0.30:
            return None

        return R, t, len(inl), inl

    # ── legacy monocular helper (kept for VIOTracker compatibility) ─────────

    @staticmethod
    def intrinsics_from_rs(rs_intr) -> np.ndarray:
        return _intr_to_K(rs_intr)
