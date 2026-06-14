import cv2
import numpy as np

def draw_dashed_rectangle(img, pt1, pt2, color, thickness=1, gap=6):
    """Draws a dashed bounding box around an object."""
    xmin, ymin = pt1
    xmax, ymax = pt2
    for x in range(xmin, xmax, gap * 2):
        cv2.line(img, (x, ymin), (min(x + gap, xmax), ymin), color, thickness)
    for x in range(xmin, xmax, gap * 2):
        cv2.line(img, (x, ymax), (min(x + gap, xmax), ymax), color, thickness)
    for y in range(ymin, ymax, gap * 2):
        cv2.line(img, (xmin, y), (xmin, min(y + gap, ymax)), color, thickness)
    for y in range(ymin, ymax, gap * 2):
        cv2.line(img, (xmax, y), (xmax, min(y + gap, ymax)), color, thickness)

def draw_dashed_line(img, pt1, pt2, color, thickness=1, gap=5):
    """Draws a dashed radar tracking line from camera center to obstacle centroid."""
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    dist = np.sqrt(dx**2 + dy**2)
    if dist == 0:
        return
    num_steps = int(dist / gap)
    if num_steps <= 1:
        cv2.line(img, pt1, pt2, color, thickness)
        return
    xs = np.linspace(pt1[0], pt2[0], num_steps)
    ys = np.linspace(pt1[1], pt2[1], num_steps)
    for i in range(0, len(xs) - 1, 2):
        start = (int(xs[i]), int(ys[i]))
        end = (int(xs[i+1]), int(ys[i+1]))
        cv2.line(img, start, end, color, thickness)
