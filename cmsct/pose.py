import cv2

def draw_skeleton(image, keypoints):
    """Draw all keypoints and skeleton joints in different colors (COCO format: 0-16)."""
    # Define colors for keypoints (17 colors in BGR format)
    keypoint_colors = [
        (255, 0, 0),    # 0: nose - Blue
        (0, 255, 0),    # 1: left eye - Green
        (0, 0, 255),    # 2: right eye - Red
        (255, 255, 0),  # 3: left ear - Cyan
        (255, 0, 255),  # 4: right ear - Magenta
        (0, 255, 255),  # 5: left shoulder - Yellow
        (128, 0, 0),    # 6: right shoulder - Dark blue
        (0, 128, 0),    # 7: left elbow - Dark green
        (0, 0, 128),    # 8: right elbow - Dark red
        (128, 128, 0),  # 9: left wrist - Olive
        (128, 0, 128),  # 10: right wrist - Purple
        (0, 128, 128),  # 11: left hip - Teal
        (255, 128, 0),  # 12: right hip - Orange
        (255, 0, 128),  # 13: left knee - Pink
        (128, 255, 0),  # 14: right knee - Lime
        (0, 255, 128),  # 15: left ankle - Spring green
        (128, 0, 255)   # 16: right ankle - Violet
    ]

    # Define colors for skeleton connections (12 connections)
    joint_colors = [
        (255, 0, 0),    # (16, 14) - Blue
        (0, 255, 0),    # (14, 12) - Green
        (0, 0, 255),    # (15, 13) - Red
        (255, 255, 0),  # (13, 11) - Cyan
        (255, 0, 255),  # (12, 11) - Magenta
        (0, 255, 255),  # (5, 11) - Yellow (corrected: left shoulder to left hip)
        (128, 0, 0),    # (6, 12) - Dark blue (right shoulder to right hip)
        (0, 128, 0),    # (5, 6) - Dark green (corrected: left shoulder to right shoulder)
        (0, 0, 128),    # (5, 7) - Dark red (left shoulder to left elbow)
        (128, 128, 0),  # (7, 9) - Olive (left elbow to left wrist)
        (128, 0, 128),  # (6, 8) - Purple (right shoulder to right elbow)
        (0, 128, 128)   # (8, 10) - Teal (right elbow to right wrist)
    ]

    # Corrected skeleton connections (COCO format: 0-16)
    skeleton = [
        (16, 14), (14, 12),  # Right leg: ankle-knee, knee-hip
        (15, 13), (13, 11),  # Left leg: ankle-knee, knee-hip
        (12, 11),            # Hips: right hip-left hip
        (5, 11), (6, 12),    # Shoulders to hips: left shoulder-left hip, right shoulder-right hip
        (5, 6),              # Shoulders: left-right
        (5, 7), (7, 9),      # Left arm: shoulder-elbow, elbow-wrist
        (6, 8), (8, 10)      # Right arm: shoulder-elbow, elbow-wrist
    ]

    # Draw all keypoints (circles)
    for idx in range(min(17, keypoints.shape[0])):  # Ensure we don’t exceed keypoint array size
        if keypoints[idx][0] > 0:  # Check if keypoint is visible (x > 0)
            pos = tuple(map(int, keypoints[idx][:2]))
            cv2.circle(image, pos, 5, keypoint_colors[idx], -1)  # Filled circle, radius 5

    # Draw skeleton joints (lines)
    for i, connection in enumerate(skeleton):
        start_idx, end_idx = connection
        if (start_idx < keypoints.shape[0] and end_idx < keypoints.shape[0] and 
            keypoints[start_idx][0] > 0 and keypoints[end_idx][0] > 0):  # Validate indices and visibility
            start = tuple(map(int, keypoints[start_idx][:2]))
            end = tuple(map(int, keypoints[end_idx][:2]))
            cv2.line(image, start, end, joint_colors[i], 2)  # Line thickness 2