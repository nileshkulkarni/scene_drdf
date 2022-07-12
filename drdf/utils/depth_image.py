import cv2
import numpy as np


def convert_depth_image_to_colormap(depth_img, max_depth=10):
    """Convert depth image to a colormap.
    Args:
        depth_img: (H, W)
        max_depth: float
    """
    depth_img = np.clip(depth_img, 0, max_depth)
    depth_img = depth_img / max_depth
    depth_img = depth_img * 255
    depth_img = np.uint8(depth_img)
    depth_img = cv2.applyColorMap(depth_img, cv2.COLORMAP_PLASMA)
    return depth_img
