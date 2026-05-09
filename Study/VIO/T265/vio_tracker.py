"""
VIO Tracker: IMU (EKF predict) + Stereo VO (EKF update, metric scale).

Flow per poll cycle:
  1. Buffer IMU samples.
  2. On stereo frame pair: integrate IMU into EKF, then apply metric visual update.
"""

import numpy as np
from ekf            import EKF
from imu_integrator import ImuIntegrator
from visual_tracker import StereoVisualTracker


class VIOTracker:
    """
    Call process_bundle(bundle) each iteration.
    """

    # Collect this many accel samples before EKF starts (≈ 1 s at 62 Hz)
    _GRAVITY_INIT_N = 60

    def __init__(self, calib=None, ekf_cfg=None, vis_cfg=None):
        self.ekf     = EKF(ekf_cfg)
        self.imu_int = ImuIntegrator()
        self.vis     = StereoVisualTracker(calib=calib, cfg=vis_cfg)

        self._frame_count  = 0
        self._vis_count    = 0
        self._last_vis_ts  = None
        self._gravity_buf  = []   # accumulates accel for initial gravity alignment
        self._gyro_buf     = []   # accumulates gyro for initial bias estimation
        self._gravity_done = False

    # ── main entry point ───────────────────────────────────────────────────

    def process_bundle(self, bundle) -> dict:
        self._maybe_init_gravity(bundle)
        self._buffer_imu(bundle)

        if bundle.has_stereo and self._gravity_done:
            self._frame_count += 1
            self.imu_int.integrate_into(self.ekf, bundle.image_ts)

            result = self.vis.process(bundle.image, bundle.image2)
            if result is not None:
                R_vis, t_metric, n_inliers = result
                t_norm = np.linalg.norm(t_metric)
                dt = float(np.clip(
                    (bundle.image_ts - self._last_vis_ts) if self._last_vis_ts else 1/30,
                    1e-3, 0.5))
                v_norm = t_norm / dt
                if v_norm > 3.0:
                    # Implausible velocity: visual tracker state is corrupted.
                    # Reset so next frame starts fresh with clean triangulation.
                    self.vis.reset()
                elif t_norm > 0.003:
                    self._visual_update(R_vis, t_metric, bundle.image_ts)
                    self._vis_count += 1

            # ZUPT runs last: overrides noisy visual drift when truly stationary
            self._zupt(bundle)

        return self._state_dict()

    def reset(self):
        self.ekf     = EKF()
        self.imu_int.reset()
        self.vis.reset()
        self._frame_count  = 0
        self._vis_count    = 0
        self._last_vis_ts  = None
        self._gravity_buf  = []
        self._gyro_buf     = []
        self._gravity_done = False

    # ── internal ───────────────────────────────────────────────────────────

    # ZUPT thresholds: |a_body| ≈ 9.81 at rest (gravity only), |ω| ≈ 0
    _ZUPT_ACCEL_THR = 0.8   # m/s² deviation from 9.81
    _ZUPT_GYRO_THR  = 0.08  # rad/s

    def _maybe_init_gravity(self, bundle):
        """Collect accel samples until we have enough to align with gravity."""
        if self._gravity_done:
            return
        for _, a in bundle.accel_samples:
            self._gravity_buf.append(a)
        for _, g in bundle.gyro_samples:
            self._gyro_buf.append(g)
        if len(self._gravity_buf) >= self._GRAVITY_INIT_N:
            self.ekf.initialize_from_accel(self._gravity_buf, self._gyro_buf)
            self.imu_int.reset()   # discard pre-alignment IMU samples
            self._gravity_done = True
            bg = self.ekf.x[13:16]
            print(f"[VIOTracker] gravity alignment done, gyro_bias={np.round(bg,5)}")

    def _zupt(self, bundle):
        """
        Zero Velocity Update: if IMU shows the device is stationary,
        inject v=0 into the EKF to cancel accumulated double-integration drift.
        """
        if not bundle.accel_samples or not bundle.gyro_samples:
            return
        a_mag = np.linalg.norm(np.mean([a for _, a in bundle.accel_samples], axis=0))
        g_mag = np.linalg.norm(np.mean([g for _, g in bundle.gyro_samples],  axis=0))
        if abs(a_mag - 9.81) < self._ZUPT_ACCEL_THR and g_mag < self._ZUPT_GYRO_THR:
            self.ekf.update_zupt()

    def _buffer_imu(self, bundle):
        if not bundle.accel_samples or not bundle.gyro_samples:
            return
        gyro_dict = {ts: g for ts, g in bundle.gyro_samples}
        gyro_ts   = sorted(gyro_dict)
        for ts_a, a in bundle.accel_samples:
            ts_g = min(gyro_ts, key=lambda t: abs(t - ts_a))
            if abs(ts_g - ts_a) < 0.01:
                self.imu_int.push(ts_a, a, gyro_dict[ts_g])

    def _visual_update(self, R_vis, t_metric, img_ts):
        """
        t_metric is the camera translation in the previous camera frame (metric).
        Convert to world frame and use as velocity measurement.
        """
        dt = (img_ts - self._last_vis_ts) if self._last_vis_ts is not None else 1 / 30
        dt = float(np.clip(dt, 1e-3, 0.5))
        self._last_vis_ts = img_ts

        # Rotate metric translation into world frame
        R_world   = self.ekf.rotation_matrix
        delta_p_w = R_world @ t_metric          # world-frame displacement

        # Visual velocity estimate → EKF velocity measurement (may be gated)
        v_vis = delta_p_w / dt
        accepted = self.ekf.update_visual(v_vis)
        if not accepted:
            self._vis_count -= 1   # undo the increment in process_bundle

    def _state_dict(self):
        return {
            "position":       self.ekf.position,
            "velocity":       self.ekf.velocity,
            "quaternion":     self.ekf.quaternion,
            "rotation":       self.ekf.rotation_matrix,
            "frame_count":    self._frame_count,
            "visual_updates": self._vis_count,
            "imu_buffered":   self.imu_int.buffered,
            "scale":          1.0,
            "vis_debug":      self.vis.last_debug,
        }
