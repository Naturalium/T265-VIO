"""
T265 stream reader via pyrealsense2 v2.51.1 (last version with T265 support).

Produces:
  - fisheye1 frames  (800×848, FISHEYE, 30 Hz)
  - IMU samples      (accel @ 62 Hz, gyro @ 200 Hz)

Uses the sensor API directly (not pipeline) because T265 requires both
fisheye sensors to be opened simultaneously.
"""

import os
import queue
import threading
import time
import numpy as np

# librealsense2 v2.51.1 빌드본을 명시적으로 로드 (pip 2.57.7은 T265 미지원)
_LRS_BUILD = os.path.expanduser("~/Project/librealsense-t265/build")
if _LRS_BUILD not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = _LRS_BUILD + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    import ctypes
    try:
        ctypes.CDLL(os.path.join(_LRS_BUILD, "librealsense2.so.2.51"))
    except OSError:
        pass

try:
    import pyrealsense2 as rs
    RS_AVAILABLE = bool(rs.context().query_devices())
    if not RS_AVAILABLE:
        print("[T265Reader] pyrealsense2 로드됨, 하지만 연결된 디바이스 없음 → MOCK 모드")
except ImportError:
    RS_AVAILABLE = False
    print("[T265Reader] pyrealsense2 not found – MOCK 모드")


class T265Reader:
    """
    Context-manager wrapper around a T265 sensor.

    Usage:
        with T265Reader() as reader:
            for bundle in reader.poll():
                ...
    """

    def __init__(self):
        if not RS_AVAILABLE:
            self._mock = True
            return
        self._mock    = False
        self._sensor  = None
        self._q       = queue.Queue(maxsize=60)
        self._intrinsics = None

    # ── context manager ────────────────────────────────────────────────────

    def __enter__(self):
        if self._mock:
            return self

        ctx = rs.context()
        dev = ctx.query_devices()[0]
        self._sensor = dev.query_sensors()[0]

        profiles = self._sensor.get_stream_profiles()
        selected = []
        seen = set()
        for p in profiles:
            st  = p.stream_type()
            key = (st, p.stream_index())
            if key in seen:
                continue
            seen.add(key)
            if st in (rs.stream.fisheye, rs.stream.gyro, rs.stream.accel):
                selected.append(p)

        # 내장 포즈는 사용하지 않고 우리 EKF로 추정
        self._sensor.open(selected)
        self._sensor.start(self._on_frame)

        # fisheye1 intrinsics 저장
        for p in selected:
            if p.stream_type() == rs.stream.fisheye and p.stream_index() == 1:
                self._intrinsics = p.as_video_stream_profile().get_intrinsics()
                break

        print(f"[T265Reader] 스트리밍 시작: {[str(p.stream_type()) for p in selected]}")
        return self

    def __exit__(self, *_):
        if not self._mock and self._sensor:
            self._sensor.stop()
            self._sensor.close()
            print("[T265Reader] 스트리밍 종료")

    # ── streaming ──────────────────────────────────────────────────────────

    def poll(self, timeout: float = 0.1):
        """
        Generator that yields FrameBundle objects indefinitely.
        Collects all frames queued since last yield into one bundle.
        """
        if self._mock:
            yield from self._mock_stream()
            return

        frame_period = 1 / 30  # fisheye 30 fps
        while True:
            bundle = FrameBundle()
            deadline = time.time() + frame_period
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    f = self._q.get(timeout=min(remaining, 0.005))
                    self._dispatch(f, bundle)
                except queue.Empty:
                    if time.time() >= deadline:
                        break
            yield bundle

    def get_fisheye_intrinsics(self):
        """Returns pyrealsense2 intrinsics object for fisheye1."""
        return self._intrinsics

    def get_stereo_calibration(self):
        """
        Returns stereo calibration dict or None (hardware only).
        Keys: intr1, intr2, R_stereo, t_stereo
        """
        if self._mock or self._sensor is None:
            return None
        profiles = self._sensor.get_stream_profiles()
        fe1 = next(p for p in profiles
                   if p.stream_type() == rs.stream.fisheye and p.stream_index() == 1)
        fe2 = next(p for p in profiles
                   if p.stream_type() == rs.stream.fisheye and p.stream_index() == 2)
        ex = fe1.as_video_stream_profile().get_extrinsics_to(
             fe2.as_video_stream_profile())
        return {
            "intr1":    fe1.as_video_stream_profile().get_intrinsics(),
            "intr2":    fe2.as_video_stream_profile().get_intrinsics(),
            "R_stereo": np.array(ex.rotation).reshape(3, 3),
            "t_stereo": np.array(ex.translation),
        }

    # ── internal ───────────────────────────────────────────────────────────

    def _on_frame(self, frame):
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            pass  # 처리 속도보다 빠르면 드롭

    def _dispatch(self, f, bundle: "FrameBundle"):
        st  = f.get_profile().stream_type()
        idx = f.get_profile().stream_index()
        ts  = f.get_timestamp() * 1e-3   # ms → s

        if st == rs.stream.fisheye:
            img = np.asanyarray(f.get_data()).copy()
            if idx == 1:
                bundle.image    = img
                bundle.image_ts = ts
            elif idx == 2:
                bundle.image2   = img

        elif st == rs.stream.accel:
            d = f.as_motion_frame().get_motion_data()
            bundle.accel_samples.append((ts, np.array([d.x, d.y, d.z])))

        elif st == rs.stream.gyro:
            d = f.as_motion_frame().get_motion_data()
            bundle.gyro_samples.append((ts, np.array([d.x, d.y, d.z])))

    # ── mock mode ──────────────────────────────────────────────────────────

    @staticmethod
    def _mock_stream():
        rng    = np.random.default_rng(42)
        t      = 0.0
        dt_img = 1 / 30
        dt_imu = 1 / 200

        while True:
            bundle = FrameBundle()
            bundle.image    = rng.integers(50, 200, (848, 800), dtype=np.uint8)
            bundle.image2   = rng.integers(50, 200, (848, 800), dtype=np.uint8)
            bundle.image_ts = t
            for k in range(7):
                ts = t - dt_img + k * dt_imu
                bundle.accel_samples.append((ts, np.array([0.0, 0.0, 9.81]) + rng.normal(0, 0.02, 3)))
                bundle.gyro_samples.append( (ts, rng.normal(0, 0.005, 3)))
            t += dt_img
            time.sleep(dt_img)
            yield bundle


class FrameBundle:
    def __init__(self):
        self.image:         np.ndarray | None = None   # fisheye1 (left)
        self.image2:        np.ndarray | None = None   # fisheye2 (right)
        self.image_ts:      float = 0.0
        self.accel_samples: list  = []
        self.gyro_samples:  list  = []

    @property
    def has_image(self) -> bool:
        return self.image is not None

    @property
    def has_stereo(self) -> bool:
        return self.image is not None and self.image2 is not None
