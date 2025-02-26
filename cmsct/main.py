# processing.py
from cmsct.detector import Detector
from cmsct.constants import PERSONDET_TRT_MODEL, PERSONDET_EXP_FILE, TrackerArgs, POSE_MODEL_PATH
from cmsct.reader import VideoReader
from cmsct.pose import draw_skeleton  # Assuming this is your corrected draw_skeleton function
from yolox.tracker.byte_tracker import BYTETracker
from argparse import Namespace 
from yolox.utils.visualize import plot_tracking
import os
import cv2
import numpy as np
from ultralytics import YOLO

def process_video(video_path):
    # Initialize video reader, detector, tracker, and pose model
    reader = VideoReader(video_path, chunk_size=60)
    detector = Detector(PERSONDET_EXP_FILE, PERSONDET_TRT_MODEL)
    args = Namespace(**TrackerArgs)
    tracker = BYTETracker(args)
    pose_model = YOLO(POSE_MODEL_PATH)
    
    frame_id = 0
    
    while True:
        frames, done = reader.read_chunk()
        if not frames or done:
            break
            
        all_outputs = detector.inference_batch(frames)
        for frame, out in zip(frames, all_outputs): 
            frame_id += 1
            if out is not None:
                online_targets = tracker.update(out, [reader.height, reader.width], detector.exp.test_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                online_img = plot_tracking(frame, online_tlwhs, online_ids, frame_id=frame_id)
            else:
                online_img = np.copy(frame)

            # Run pose estimation
            pose_results = pose_model.track(frame, persist=True, verbose=False)
            if pose_results[0].boxes.id is not None:
                keypoints = pose_results[0].keypoints.xy.cpu().numpy()
                for kpts in keypoints:
                    draw_skeleton(online_img, kpts)

            # Yield the processed frame
            yield online_img

    reader.release()  # Assuming VideoReader has a release method

