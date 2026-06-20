import numpy as np

def detect_lanes(frame: np.ndarray):
    """Simple lane detection placeholder.

    Returns four points defining the left and right lane boundaries.
    This is a heuristic based on the frame dimensions and works as a fallback
    when a dedicated lane detection model is not available.

    Args:
        frame (np.ndarray): BGR image frame.
    Returns:
        tuple: (pt_left_bottom, pt_left_top, pt_right_bottom, pt_right_top)
    """
    h, w = frame.shape[:2]
    left_x_bottom = int(w * 0.35)
    left_x_top = int(w * 0.42)
    right_x_bottom = int(w * 0.98)
    right_x_top = int(w * 0.65)

    top_y = int(h * 0.45)
    bottom_y = h
    pt_left_bottom = (left_x_bottom, bottom_y)
    pt_left_top = (left_x_top, top_y)
    pt_right_bottom = (right_x_bottom, bottom_y)
    pt_right_top = (right_x_top, top_y)
    return pt_left_bottom, pt_left_top, pt_right_bottom, pt_right_top


