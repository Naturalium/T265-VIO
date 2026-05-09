"""
High-frequency IMU integration buffer.
Buffers raw IMU samples between visual frames and feeds them to the EKF.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass


@dataclass
class ImuSample:
    timestamp: float   # seconds
    accel: np.ndarray  # [3] m/s²
    gyro: np.ndarray   # [3] rad/s


class ImuIntegrator:
    """
    Accumulates IMU samples and integrates them into the EKF at each
    visual frame, preserving sub-frame timing.
    """

    def __init__(self):
        self._buf: deque[ImuSample] = deque()
        self._last_ts: float | None = None

    def push(self, ts: float, accel: np.ndarray, gyro: np.ndarray):
        self._buf.append(ImuSample(ts, accel.copy(), gyro.copy()))

    def integrate_into(self, ekf, up_to_ts: float):
        """
        Run EKF.predict() for all buffered samples with timestamp ≤ up_to_ts.
        Returns the number of samples consumed.
        """
        count = 0
        while self._buf and self._buf[0].timestamp <= up_to_ts:
            s = self._buf.popleft()
            if self._last_ts is not None:
                dt = s.timestamp - self._last_ts
                if 1e-6 < dt < 0.5:           # sanity: 1µs … 500ms
                    ekf.predict(s.accel, s.gyro, dt)
            self._last_ts = s.timestamp
            count += 1
        return count

    def reset(self):
        self._buf.clear()
        self._last_ts = None

    @property
    def buffered(self) -> int:
        return len(self._buf)

    # ── static calibration helpers ─────────────────────────────────────────

    @staticmethod
    def calibrate_bias(samples: list[ImuSample]):
        """
        Estimate static accel/gyro bias from a list of stationary samples.
        Returns (accel_bias, gyro_bias) as numpy arrays [3].
        """
        accels = np.stack([s.accel for s in samples])
        gyros  = np.stack([s.gyro  for s in samples])

        g_ref = np.array([0, 0, 9.81])    # expected gravity vector
        accel_bias = accels.mean(axis=0) - g_ref
        gyro_bias  = gyros.mean(axis=0)

        accel_std = accels.std(axis=0)
        gyro_std  = gyros.std(axis=0)

        print(f"[IMU calib] accel bias: {accel_bias}, std: {accel_std}")
        print(f"[IMU calib] gyro  bias: {gyro_bias},  std: {gyro_std}")
        return accel_bias, gyro_bias
