"""
T265 VIO Tracker — main entry point.

Usage:
    python main.py           # live T265 + visualization
    python main.py --no-vis  # headless (terminal output only)
    python main.py --mock    # force mock data
"""

import argparse
import time
import sys

from t265_reader    import T265Reader
from vio_tracker    import VIOTracker


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mock",   action="store_true", help="Force synthetic data")
    p.add_argument("--no-vis", action="store_true", help="Headless mode (no window)")
    return p.parse_args()


def main():
    args  = parse_args()
    show  = not args.no_vis

    tracker = VIOTracker()
    t_start = time.time()

    if show:
        from visualizer import Visualizer
        viz = Visualizer("T265 VIO — EKF Tracker")

    with T265Reader() as reader:
        if not args.mock:
            calib = reader.get_stereo_calibration()
            if calib is not None:
                tracker = VIOTracker(calib=calib)

        print("Running — press 'q' to quit.")
        last_frame  = None
        last_frame2 = None

        for bundle in reader.poll():
            state = tracker.process_bundle(bundle)
            if bundle.has_image:
                last_frame  = bundle.image
            if bundle.image2 is not None:
                last_frame2 = bundle.image2

            # Terminal log every 30 video frames
            if state["frame_count"] % 30 == 0 and state["frame_count"] > 0:
                elapsed = time.time() - t_start
                p = state["position"]
                print(f"[{elapsed:6.1f}s] frames={state['frame_count']:4d} "
                      f"vis={state['visual_updates']:3d} "
                      f"pos=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})")

            if show:
                viz.update(last_frame, state, frame2=last_frame2)
                if viz.wait_key(1) == ord('q'):
                    break

    if show:
        viz.close()
    print("Done.")


if __name__ == "__main__":
    main()
