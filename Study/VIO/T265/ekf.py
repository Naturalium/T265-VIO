"""
Extended Kalman Filter for Visual-Inertial Odometry.

State vector (16-dim):
  [0:3]  position     (p_x, p_y, p_z)
  [3:6]  velocity     (v_x, v_y, v_z)
  [6:10] quaternion   (q_w, q_x, q_y, q_z)   -- world-from-body
  [10:13] accel bias  (b_ax, b_ay, b_az)
  [13:16] gyro  bias  (b_gx, b_gy, b_gz)
"""

import numpy as np


# ── quaternion helpers ──────────────────────────────────────────────────────

def quat_mult(q, r):
    """Hamilton product q ⊗ r, both [w,x,y,z]."""
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = r
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1,
    ])


def quat_norm(q):
    return q / np.linalg.norm(q)


def quat_to_rot(q):
    """Rotation matrix R from quaternion [w,x,y,z]."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),   2*(x*z + w*y)],
        [2*(x*y + w*z),   1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def skew(v):
    """3×3 skew-symmetric matrix of vector v."""
    x, y, z = v
    return np.array([
        [ 0, -z,  y],
        [ z,  0, -x],
        [-y,  x,  0],
    ])


# ── EKF ────────────────────────────────────────────────────────────────────

class EKF:
    """
    Loosely-coupled EKF for VIO.

    Process noise (Q):
      - accel noise, gyro noise, accel bias walk, gyro bias walk

    Measurement noise (R):
      - 6-DOF relative pose from visual odometry [dx,dy,dz, dRx,dRy,dRz]
    """

    N = 16  # state dimension

    def __init__(self, cfg=None):
        cfg = cfg or {}

        # Initial state
        self.x = np.zeros(self.N)
        self.x[6] = 1.0  # q_w = 1  (identity quaternion)

        # Initial covariance — keep tight so Kalman gain starts small.
        self.P = np.diag([
            *([5e-3]*3),   # position
            *([5e-2]*3),   # velocity
            *([1e-4]*4),   # quaternion
            *([1e-3]*3),   # accel bias
            *([1e-4]*3),   # gyro bias
        ])

        # Process noise spectral densities.
        # T265 IMU specs: accel ~0.008 m/s²/√Hz, gyro ~0.002 rad/s/√Hz.
        # Values slightly above spec to account for vibration/temp drift.
        self.sigma_a  = cfg.get("sigma_a",  0.01)   # accel noise  [m/s²/√Hz]
        self.sigma_g  = cfg.get("sigma_g",  0.002)  # gyro noise   [rad/s/√Hz]
        self.sigma_ba = cfg.get("sigma_ba", 5e-5)   # accel bias walk
        self.sigma_bg = cfg.get("sigma_bg", 1e-5)   # gyro bias walk

        # Velocity measurement noise (m/s std dev).
        # Visual translation Δp has ~5-10 mm noise at 30 fps → v = Δp/dt ≈ 0.15-0.3 m/s noise.
        r_vel = cfg.get("r_vel", 0.3) ** 2
        self.R_vel = np.diag([r_vel] * 3)

        # Chi-squared gate threshold for visual update (3-DOF, 99.9% → 16.27).
        self._vis_gate = cfg.get("vis_gate", 16.0)

    def initialize_from_accel(self, accel_samples, gyro_samples=None):
        """
        Set initial quaternion from gravity direction and pre-set gyro bias.
        Call once with ~1 s of stationary IMU data.
        """
        g_body = np.mean(accel_samples, axis=0)
        g_hat  = g_body / (np.linalg.norm(g_body) + 1e-9)
        z_world = np.array([0., 0., 1.])

        axis = np.cross(g_hat, z_world)
        sin_a = np.linalg.norm(axis)

        if sin_a < 1e-6:
            q = np.array([1., 0., 0., 0.]) if g_hat[2] > 0 else np.array([0., 1., 0., 0.])
        else:
            axis /= sin_a
            angle = np.arctan2(sin_a, np.dot(g_hat, z_world))
            q = np.r_[np.cos(angle / 2), np.sin(angle / 2) * axis]

        self.x[6:10] = quat_norm(q)

        # Pre-set gyro bias from static measurement (static gyro reading ≈ bias)
        if gyro_samples is not None and len(gyro_samples) > 0:
            self.x[13:16] = np.mean(gyro_samples, axis=0)

    def update_zupt(self):
        """
        Zero Velocity Update with tight noise (σ = 0.01 m/s).
        Use when IMU confirms device is stationary.
        """
        R_zupt = np.eye(3) * (0.01 ** 2)
        H = np.zeros((3, self.N))
        H[0:3, 3:6] = np.eye(3)
        y = -self.x[3:6]   # innovation = 0 - v_current
        S = H @ self.P @ H.T + R_zupt
        K = self.P @ H.T @ np.linalg.solve(S, np.eye(3)).T
        dx = K @ y
        self.x    += dx
        self.x[6:10] = quat_norm(self.x[6:10])
        I_KH   = np.eye(self.N) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_zupt @ K.T

    # ── prediction ─────────────────────────────────────────────────────────

    def predict(self, accel_raw, gyro_raw, dt):
        """
        IMU prediction step.
        accel_raw, gyro_raw: numpy arrays [3] in body frame [m/s², rad/s].
        """
        p = self.x[0:3]
        v = self.x[3:6]
        q = self.x[6:10]
        ba = self.x[10:13]
        bg = self.x[13:16]

        R = quat_to_rot(q)
        a_body = accel_raw - ba           # bias-corrected accel (body frame)
        w_body = gyro_raw  - bg           # bias-corrected gyro  (body frame)
        g      = np.array([0, 0, -9.81])  # gravity (world frame)

        a_world = R @ a_body + g

        # ── state integration (midpoint / 1st order) ──
        p_new = p + v * dt + 0.5 * a_world * dt**2
        v_new = v + a_world * dt

        # quaternion update: q ← q ⊗ exp(0.5 * w * dt)
        angle = np.linalg.norm(w_body) * dt
        if angle > 1e-8:
            axis = w_body / np.linalg.norm(w_body)
            dq = np.r_[np.cos(angle/2), np.sin(angle/2) * axis]
        else:
            dq = np.array([1.0, *( 0.5 * w_body * dt)])
        q_new = quat_norm(quat_mult(q, dq))

        self.x[0:3]  = p_new
        self.x[3:6]  = v_new
        self.x[6:10] = q_new
        # biases unchanged

        # ── linearised F matrix ─────────────────────────────────────────
        F = np.eye(self.N)

        # dp/dv
        F[0:3, 3:6] = np.eye(3) * dt
        # dp/dq  (via da_world/dq applied twice)
        F[0:3, 6:10] = 0.5 * self._da_dq(a_body, q) * dt**2
        # dv/dq
        F[3:6, 6:10] = self._da_dq(a_body, q) * dt
        # dp/dba, dv/dba
        F[0:3, 10:13] = -0.5 * R * dt**2
        F[3:6, 10:13] = -R * dt
        # dq/dbg
        F[6:10, 13:16] = self._dq_dbg(q, dt)

        # ── process noise Q ─────────────────────────────────────────────
        Q = np.zeros((self.N, self.N))
        sa2 = self.sigma_a**2 * dt
        sg2 = self.sigma_g**2 * dt
        Q[3:6,  3:6]  = np.eye(3) * sa2       # velocity noise
        Q[0:3,  0:3]  = np.eye(3) * sa2 * dt**2 * 0.25
        Q[6:10, 6:10] = np.eye(4) * sg2 * 0.25
        Q[10:13,10:13] = np.eye(3) * self.sigma_ba**2 * dt
        Q[13:16,13:16] = np.eye(3) * self.sigma_bg**2 * dt

        self.P = F @ self.P @ F.T + Q

    # ── visual update ───────────────────────────────────────────────────────

    def update_visual(self, v_vis):
        """
        EKF update using visual velocity estimate.

        v_vis: velocity [3] in world frame (m/s).
               Computed as (scaled visual displacement) / dt.

        Returns True if the measurement was accepted, False if gated out.
        """
        H = np.zeros((3, self.N))
        H[0:3, 3:6] = np.eye(3)    # measures velocity state

        y      = v_vis - self.x[3:6]
        R_meas = self.R_vel
        S      = H @ self.P @ H.T + R_meas

        # Mahalanobis distance gate: rejects visually implausible velocity jumps.
        # chi2(3, 99.9%) = 16.27 — a measurement beyond this is a statistical
        # outlier (bad PnP frame) and should not be fused.
        md2 = float(y @ np.linalg.solve(S, y))
        if md2 > self._vis_gate:
            return False

        K = self.P @ H.T @ np.linalg.solve(S, np.eye(3)).T

        dx = K @ y
        self.x    += dx
        self.x[6:10] = quat_norm(self.x[6:10])

        I_KH   = np.eye(self.N) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_meas @ K.T  # Joseph form
        return True

    # ── accessors ──────────────────────────────────────────────────────────

    @property
    def position(self):
        return self.x[0:3].copy()

    @property
    def velocity(self):
        return self.x[3:6].copy()

    @property
    def quaternion(self):
        return self.x[6:10].copy()

    @property
    def rotation_matrix(self):
        return quat_to_rot(self.x[6:10])

    # ── internal Jacobians ─────────────────────────────────────────────────

    def _da_dq(self, a_body, q):
        """∂(R(q) a_body) / ∂q  →  shape (3, 4)."""
        w, x, y, z = q
        ax, ay, az = a_body
        # Analytical derivative of R(q)*a w.r.t. [w,x,y,z]
        J = 2 * np.array([
            [ w*ax - z*ay + y*az,  x*ax + y*ay + z*az, -y*ax + x*ay + w*az, -z*ax - w*ay + x*az],
            [ z*ax + w*ay - x*az,  y*ax - x*ay - w*az,  x*ax + y*ay + z*az,  w*ax - z*ay + y*az],
            [-y*ax + x*ay + w*az,  z*ax + w*ay - x*az, -w*ax + z*ay - y*az,  x*ax + y*ay + z*az],
        ])
        return J

    def _dq_dbg(self, q, dt):
        """∂q_new / ∂b_g  →  shape (4, 3), for small gyro bias correction."""
        w, x, y, z = q
        return -0.5 * dt * np.array([
            [-x, -y, -z],
            [ w, -z,  y],
            [ z,  w, -x],
            [-y,  x,  w],
        ])
