"""
Real-time VIO visualizer using OpenCV.

Layout (1280 × 720):
  ┌────────────────────────────────┬───────────┬───────────┐
  │  Stereo Debug View             │ 3D Axes   │ Art. Hrz  │
  │  Left (400) | Right (400)      │  240×300  │  240×300  │  300 px
  │  with tracks + stereo matches  │           │           │
  ├────────────────────┬───────────┴───────────┴───────────┤
  │  XY traj  640×200  │  XZ traj            640×200       │  200 px
  ├────────────────────┴──────────────────────────────────  ┤
  │  RPY graph                             1280×110         │  110 px
  ├───────────────────────────────────────────────────────  ┤
  │  Position graph                        1280×110         │  110 px
  └───────────────────────────────────────────────────────  ┘
  Total: 1280 × 720
"""

import cv2
import numpy as np
from collections import deque


_BG    = (20,  20,  20)
_RED   = (60,  60, 220)
_GREEN = (60, 200,  60)
_BLUE  = (220,  60,  60)
_WHITE = (220, 220, 220)
_GRAY  = (100, 100, 100)
_CYAN  = (200, 200,  40)
_AMBER = (40,  170, 230)
_YELLOW= (40,  220, 220)
_PINK  = (180,  80, 200)


def _quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def _rot_to_euler(R):
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        roll  = np.degrees(np.arctan2( R[2,1], R[2,2]))
        pitch = np.degrees(np.arctan2(-R[2,0], sy))
        yaw   = np.degrees(np.arctan2( R[1,0], R[0,0]))
    else:
        roll  = np.degrees(np.arctan2(-R[1,2], R[1,1]))
        pitch = np.degrees(np.arctan2(-R[2,0], sy))
        yaw   = 0.0
    return roll, pitch, yaw


def _depth_color(depth, z_min=0.15, z_max=2.0):
    """Map depth [m] → BGR colour (blue=near, green=mid, red=far)."""
    t = np.clip((depth - z_min) / (z_max - z_min), 0.0, 1.0)
    # blue → cyan → green → yellow → red
    hue = int((1.0 - t) * 120)   # 120° (green) … 0° (red) → for near: 120→240 (blue)
    hue = int((1.0 - t) * 240)   # near=blue(240), far=red(0)
    hue = max(0, min(179, hue))
    hsv = np.uint8([[[hue, 230, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


# ── Stereo debug panel ──────────────────────────────────────────────────────

def _draw_stereo_panel(dbg: dict, panel_w=800, panel_h=300) -> np.ndarray:
    """
    Returns (panel_h, panel_w, 3) BGR image with:
      - Left undistorted image (left half)
      - Right undistorted image (right half)
      - Yellow: temporally tracked features (prev → curr in left)
      - Green:  PnP inliers among tracked features
      - Coloured dots + lines: stereo-matched features (colour = depth)
      - Horizontal dashed epipolar reference lines (gray)
    """
    half = panel_w // 2
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    def _place(img, x_off):
        if img is None:
            return
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bgr  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        rsz  = cv2.resize(bgr, (half, panel_h))
        panel[:, x_off:x_off+half] = rsz

    und_l = dbg.get('und_l')
    und_r = dbg.get('und_r')
    _place(und_l, 0)
    _place(und_r, half)

    # Scale factors: original image → panel cell
    if und_l is not None:
        oh, ow = und_l.shape[:2]
        sx = half / ow
        sy = panel_h / oh
    else:
        sx = sy = 1.0

    def _lpt(pt):
        return (int(pt[0] * sx), int(pt[1] * sy))

    def _rpt(pt):
        return (int(pt[0] * sx + half), int(pt[1] * sy))

    # ── Epipolar reference lines (dashed, gray) ─────────────────────────
    for row_frac in [0.25, 0.5, 0.75]:
        ry = int(row_frac * panel_h)
        for x in range(0, panel_w, 12):
            cv2.line(panel, (x, ry), (min(x+6, panel_w-1), ry), (50, 50, 50), 1)

    # ── Stereo matches: dots + connecting lines ─────────────────────────
    sl = dbg.get('stereo_l')
    sr = dbg.get('stereo_r')
    depths = dbg.get('depths')
    if sl is not None and sr is not None and depths is not None:
        # subsample for readability
        step = max(1, len(sl) // 60)
        for i in range(0, len(sl), step):
            col = _depth_color(depths[i])
            lp  = _lpt(sl[i])
            rp  = _rpt(sr[i])
            cv2.circle(panel, lp, 3, col, -1, cv2.LINE_AA)
            cv2.circle(panel, rp, 3, col, -1, cv2.LINE_AA)
            # connecting line across the divider
            cv2.line(panel, lp, rp, col, 1, cv2.LINE_AA)

    # ── Temporal tracked features ───────────────────────────────────────
    tp = dbg.get('tracked_prev')
    tc = dbg.get('tracked_curr')
    ti = dbg.get('tracked_inliers')
    if tp is not None and tc is not None:
        for k in range(len(tp)):
            is_inlier = (ti is not None and ti[k])
            col_prev = _GRAY
            col_curr = _GREEN if is_inlier else _YELLOW
            pp = _lpt(tp[k])
            pc = _lpt(tc[k])
            cv2.line(panel, pp, pc, (60, 60, 60), 1, cv2.LINE_AA)
            cv2.circle(panel, pc, 3 if is_inlier else 2, col_curr, -1, cv2.LINE_AA)

    # ── Divider line ────────────────────────────────────────────────────
    cv2.line(panel, (half, 0), (half, panel_h), _GRAY, 1)

    # ── Labels ─────────────────────────────────────────────────────────
    cv2.putText(panel, "LEFT  (undistorted)",
                (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, _WHITE, 1, cv2.LINE_AA)
    cv2.putText(panel, "RIGHT (undistorted)",
                (half+6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, _WHITE, 1, cv2.LINE_AA)

    n_stereo  = len(sl) if sl is not None else 0
    n_tracked = len(tc) if tc is not None else 0
    n_inliers = int(np.sum(ti)) if ti is not None else 0
    cv2.putText(panel,
                f"stereo:{n_stereo}  tracked:{n_tracked}  inliers:{n_inliers}",
                (6, panel_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, _AMBER, 1, cv2.LINE_AA)

    # Depth colour legend
    for i, d in enumerate([0.2, 0.5, 1.0, 1.5, 2.0]):
        col = _depth_color(d)
        lx = half + 6 + i * 48
        cv2.rectangle(panel, (lx, panel_h-18), (lx+40, panel_h-4), col, -1)
        cv2.putText(panel, f"{d:.1f}m", (lx, panel_h-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, _WHITE, 1, cv2.LINE_AA)

    return panel


# ── 3-D axes ────────────────────────────────────────────────────────────────

def _draw_axes_3d(canvas, R, cx, cy, size=90):
    view = np.array([
        [ 0.866, -0.5,  0.0],
        [ 0.0,  0.866, -0.5],
        [ 0.0,  0.0,   0.0],
    ], dtype=float)
    for label, v, color in [('X', [1,0,0], _RED),
                              ('Y', [0,1,0], _GREEN),
                              ('Z', [0,0,1], _BLUE)]:
        vr  = R @ np.array(v, dtype=float)
        p2d = (view @ vr)[:2] * size
        x2  = int(cx + p2d[0])
        y2  = int(cy - p2d[1])
        cv2.arrowedLine(canvas, (cx, cy), (x2, y2), color, 2, tipLength=0.25)
        cv2.putText(canvas, label, (x2+4, y2+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    for v, lbl in [([1,0,0],'x'), ([0,1,0],'y'), ([0,0,1],'z')]:
        p2d = (view @ np.array(v, dtype=float))[:2] * size * 0.4
        cv2.line(canvas, (cx, cy),
                 (int(cx+p2d[0]), int(cy-p2d[1])), _GRAY, 1, cv2.LINE_AA)


# ── Artificial horizon ──────────────────────────────────────────────────────

def _draw_artificial_horizon(canvas, roll_deg, pitch_deg, cx, cy, r=100):
    cv2.circle(canvas, (cx, cy), r, (80, 60, 40), -1)
    mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r-1, 255, -1)

    pitch_px = int(pitch_deg / 90.0 * r)
    roll_rad = np.radians(roll_deg)
    cos_r, sin_r = np.cos(roll_rad), np.sin(roll_rad)

    def rot_pt(px, py):
        dx, dy = px - cx, py - cy + pitch_px
        return (int(cx + dx*cos_r - dy*sin_r),
                int(cy + dx*sin_r + dy*cos_r))

    pts_top = []
    for ang in np.linspace(0, np.pi, 60):
        px = int(cx + r * np.cos(ang + np.pi))
        py = int(cy - r * np.sin(ang + np.pi))
        pts_top.append(list(rot_pt(px, py)))
    pts_top += [[cx+r, cy], [cx-r, cy]]

    sky_canvas = canvas.copy()
    cv2.fillPoly(sky_canvas, [np.array(pts_top, dtype=np.int32)], (150, 100, 50))
    canvas[:] = np.where(mask[:, :, np.newaxis] > 0, sky_canvas, canvas)

    hw = int(r * 0.9)
    cv2.line(canvas, rot_pt(cx-hw, cy), rot_pt(cx+hw, cy), _WHITE, 2, cv2.LINE_AA)
    for deg in range(-80, 81, 10):
        if deg == 0:
            continue
        w = r//4 if deg % 30 == 0 else r//8
        cv2.line(canvas,
                 rot_pt(cx-w, cy + int(deg/90*r)),
                 rot_pt(cx+w, cy + int(deg/90*r)),
                 _WHITE, 1, cv2.LINE_AA)

    cv2.line(canvas, (cx-30, cy), (cx-10, cy), _AMBER, 3)
    cv2.line(canvas, (cx+10, cy), (cx+30, cy), _AMBER, 3)
    cv2.line(canvas, (cx,    cy-10), (cx, cy+10), _AMBER, 2)
    cv2.circle(canvas, (cx, cy), r, _GRAY, 2)
    cv2.putText(canvas, f"R {roll_deg:+.1f}",  (cx-r, cy+r+16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, _RED,   1, cv2.LINE_AA)
    cv2.putText(canvas, f"P {pitch_deg:+.1f}", (cx+10, cy+r+16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, _GREEN, 1, cv2.LINE_AA)


# ── Trajectory ──────────────────────────────────────────────────────────────

def _draw_trajectory(canvas, traj, plane, cx, cy, size, scale_m=2.0, color=_CYAN):
    if len(traj) < 2:
        return
    ppm = size / scale_m
    for d in np.arange(-scale_m, scale_m+0.1, scale_m/4):
        gx, gy = int(cx + d*ppm), int(cy + d*ppm)
        cv2.line(canvas, (gx, cy-size), (gx, cy+size), (40,40,40), 1)
        cv2.line(canvas, (cx-size, gy), (cx+size, gy), (40,40,40), 1)
    cv2.line(canvas, (cx-size,cy), (cx+size,cy), _GRAY, 1)
    cv2.line(canvas, (cx,cy-size), (cx,cy+size), _GRAY, 1)

    pts = []
    for p in list(traj)[-400:]:
        px = int(np.clip(cx + p[plane[0]]*ppm, cx-size, cx+size))
        py = int(np.clip(cy - p[plane[1]]*ppm, cy-size, cy+size))
        pts.append((px, py))
    for k in range(1, len(pts)):
        alpha = k / len(pts)
        c = tuple(int(v * alpha) for v in color)
        cv2.line(canvas, pts[k-1], pts[k], c, 1, cv2.LINE_AA)
    if pts:
        cv2.circle(canvas, pts[-1], 4, _WHITE, -1)


# ── Time-series graph ───────────────────────────────────────────────────────

def _draw_timeseries(canvas, histories, colors, labels, x0, y0, w, h,
                     y_range=(-180, 180), unit=""):
    cv2.rectangle(canvas, (x0, y0), (x0+w, y0+h), (30,30,30), -1)
    cv2.rectangle(canvas, (x0, y0), (x0+w, y0+h), _GRAY, 1)
    mid_y = y0 + h // 2
    cv2.line(canvas, (x0, mid_y), (x0+w, mid_y), (50,50,50), 1)

    span = y_range[1] - y_range[0]
    n_pts = w

    for hist, col, lbl in zip(histories, colors, labels):
        data = list(hist)[-n_pts:]
        if len(data) < 2:
            continue
        pts = []
        for k, v in enumerate(data):
            px = x0 + int(k / n_pts * w)
            frac = (v - y_range[0]) / span
            py = y0 + h - int(np.clip(frac, 0, 1) * h)
            pts.append((px, py))
        for k in range(1, len(pts)):
            cv2.line(canvas, pts[k-1], pts[k], col, 1, cv2.LINE_AA)
        if data:
            txt = f"{lbl} {data[-1]:+.2f}{unit}"
            cv2.putText(canvas, txt,
                        (x0 + 4 + labels.index(lbl)*130, y0+14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)


# ── Main Visualizer ─────────────────────────────────────────────────────────

class Visualizer:
    W, H = 1280, 720

    _STEREO_W = 800   # stereo debug panel width
    _STEREO_H = 300   # row 0 height
    _AXES_W   = 240
    _HRZ_W    = 240   # W - _STEREO_W - _AXES_W = 1280-800-240 = 240
    _TRAJ_H   = 200
    _GRAPH_H  = 110   # (720 - 300 - 200) / 2 = 110

    HISTORY = 800

    def __init__(self, window_name="T265 VIO"):
        self.win = window_name
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, self.W, self.H)

        self._canvas = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        self._traj   = deque(maxlen=2000)

        N = self.HISTORY
        self._h_roll  = deque([0.0]*N, maxlen=N)
        self._h_pitch = deque([0.0]*N, maxlen=N)
        self._h_yaw   = deque([0.0]*N, maxlen=N)
        self._h_px    = deque([0.0]*N, maxlen=N)
        self._h_py    = deque([0.0]*N, maxlen=N)
        self._h_pz    = deque([0.0]*N, maxlen=N)

    def update(self, frame, state: dict, frame2=None) -> np.ndarray:
        canvas = self._canvas
        canvas[:] = _BG

        pos  = state["position"]
        quat = state["quaternion"]
        R    = _quat_to_rot(quat)
        roll, pitch, yaw = _rot_to_euler(R)

        self._traj.append(pos.copy())
        self._h_roll.append(roll)
        self._h_pitch.append(pitch)
        self._h_yaw.append(yaw)
        self._h_px.append(pos[0])
        self._h_py.append(pos[1])
        self._h_pz.append(pos[2])

        # ── Row 0: stereo debug (left) + 3D axes + horizon (right) ────────
        dbg = state.get('vis_debug', {})

        # If we have undistorted images from the tracker, prefer them;
        # otherwise fall back to raw fisheye frames.
        if not dbg or dbg.get('und_l') is None:
            # Build a minimal dbg from raw frames
            dbg = dict(dbg or {})
            dbg.setdefault('und_l', frame)
            dbg.setdefault('und_r', frame2)

        stereo_panel = _draw_stereo_panel(dbg,
                                          panel_w=self._STEREO_W,
                                          panel_h=self._STEREO_H)
        canvas[0:self._STEREO_H, 0:self._STEREO_W] = stereo_panel

        # HUD text over stereo panel
        self._hud_text(canvas, state, roll, pitch, yaw, x0=6, y0=self._STEREO_H-80)

        # 3D axes panel
        ax_x0 = self._STEREO_W
        ax_panel = canvas[0:self._STEREO_H, ax_x0:ax_x0+self._AXES_W]
        _draw_axes_3d(ax_panel, R, self._AXES_W//2, self._STEREO_H//2, size=90)
        cv2.putText(ax_panel, "Body Frame", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _WHITE, 1, cv2.LINE_AA)
        cv2.putText(ax_panel, f"Yaw {yaw:+.1f}deg", (6, self._STEREO_H-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, _AMBER, 1, cv2.LINE_AA)

        # Artificial horizon panel
        hz_x0 = ax_x0 + self._AXES_W
        hz_panel = canvas[0:self._STEREO_H, hz_x0:self.W]
        hz_w = self.W - hz_x0
        _draw_artificial_horizon(hz_panel, roll, pitch,
                                  hz_w//2, self._STEREO_H//2, r=100)
        cv2.putText(hz_panel, "Attitude", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _WHITE, 1, cv2.LINE_AA)

        # Dividers
        cv2.line(canvas, (self._STEREO_W, 0), (self._STEREO_W, self._STEREO_H), _GRAY, 1)
        cv2.line(canvas, (ax_x0+self._AXES_W, 0),
                 (ax_x0+self._AXES_W, self._STEREO_H), _GRAY, 1)

        # ── Row 1: trajectories ────────────────────────────────────────────
        y1 = self._STEREO_H
        half = self.W // 2
        sz = 85

        cv2.putText(canvas, "XY top-down", (8, y1+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, _WHITE, 1, cv2.LINE_AA)
        _draw_trajectory(canvas, self._traj, (0,1),
                         half//2, y1+self._TRAJ_H//2, sz, scale_m=1.5, color=_CYAN)
        self._axis_labels(canvas, half//2, y1+self._TRAJ_H//2, sz, "X","Y")

        cv2.putText(canvas, "XZ side", (half+8, y1+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, _WHITE, 1, cv2.LINE_AA)
        _draw_trajectory(canvas, self._traj, (0,2),
                         half+half//2, y1+self._TRAJ_H//2, sz, scale_m=1.5, color=_AMBER)
        self._axis_labels(canvas, half+half//2, y1+self._TRAJ_H//2, sz, "X","Z")
        cv2.line(canvas, (half,y1), (half,y1+self._TRAJ_H), _GRAY, 1)

        # ── Row 2: RPY graph ───────────────────────────────────────────────
        y2 = y1 + self._TRAJ_H
        _draw_timeseries(canvas,
                         [self._h_roll, self._h_pitch, self._h_yaw],
                         [_RED, _GREEN, _BLUE],
                         ["Roll", "Pitch", "Yaw"],
                         0, y2, self.W, self._GRAPH_H,
                         y_range=(-180, 180), unit="deg")
        cv2.putText(canvas, "RPY", (self.W-50, y2+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, _GRAY, 1, cv2.LINE_AA)

        # ── Row 3: position graph ──────────────────────────────────────────
        y3 = y2 + self._GRAPH_H
        vals = [list(self._h_px)[-1], list(self._h_py)[-1], list(self._h_pz)[-1]]
        max_pos = max(0.3, max(abs(v) for v in vals) * 1.5 + 0.01)
        _draw_timeseries(canvas,
                         [self._h_px, self._h_py, self._h_pz],
                         [_RED, _GREEN, _BLUE],
                         ["X", "Y", "Z"],
                         0, y3, self.W, self._GRAPH_H,
                         y_range=(-max_pos, max_pos), unit="m")
        cv2.putText(canvas, "Pos", (self.W-50, y3+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, _GRAY, 1, cv2.LINE_AA)

        cv2.imshow(self.win, canvas)
        return canvas

    def wait_key(self, ms=1) -> int:
        return cv2.waitKey(ms) & 0xFF

    def close(self):
        cv2.destroyWindow(self.win)

    @staticmethod
    def _hud_text(canvas, state, roll, pitch, yaw, x0, y0):
        p = state["position"]
        v = state["velocity"]
        lines = [
            f"Pos  X{p[0]:+6.3f}  Y{p[1]:+6.3f}  Z{p[2]:+6.3f} m",
            f"Vel  X{v[0]:+5.2f}  Y{v[1]:+5.2f}  Z{v[2]:+5.2f} m/s",
            f"Roll {roll:+.1f}  Pitch {pitch:+.1f}  Yaw {yaw:+.1f} deg",
            f"Frames {state['frame_count']}  VisUpd {state['visual_updates']}",
        ]
        for i, txt in enumerate(lines):
            cv2.putText(canvas, txt, (x0, y0 + i*18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, _WHITE, 1, cv2.LINE_AA)

    @staticmethod
    def _axis_labels(canvas, cx, cy, size, xl, yl):
        cv2.putText(canvas, f"+{xl}", (cx+size-16, cy+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, _GRAY, 1)
        cv2.putText(canvas, f"+{yl}", (cx+4, cy-size+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, _GRAY, 1)
