import streamlit as st
import cv2
import numpy as np
import os
from loguru import logger
from cmsct.constants import PERSONDET_TRT_MODEL, PERSONDET_EXP_FILE, TrackerArgs
from cmsct.reader import VideoReader
from yolox.tracker.byte_tracker import BYTETracker
from argparse import Namespace
from yolox.utils.visualize import plot_tracking  # Assuming your modified plot_tracking is here
from cmsct.detector import Detector
# Configure loguru to capture logs in a string buffer for display
log_buffer = []
logger.remove()  # Remove default handler
logger.add(lambda msg: log_buffer.append(msg), format="{time} - {level} - {message}")

# Streamlit app
st.title("Person Detection and Tracking App")

# Video path input
video_path = st.text_input("Enter Video File Path", "dependencies/ByteTrack/videos/palace.mp4")

# Placeholder for video display
video_placeholder = st.empty()

# Placeholder for logs
log_placeholder = st.empty()

# Start button to trigger processing
if st.button("Start Processing"):
    if not os.path.exists(video_path):
        st.error("Video file not found!")
        logger.error(f"Video file not found: {video_path}")
    else:
        # Initialize save directory and components
        save_dir = "data/"
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Processing video: {video_path}")

        # Initialize video reader, detector, and tracker
        reader = VideoReader(video_path, chunk_size=60)
        detector = Detector(PERSONDET_EXP_FILE, PERSONDET_TRT_MODEL)
        args = Namespace(**TrackerArgs)
        tracker = BYTETracker(args)
        frame_id = 0

        # Process video
        while True:
            frames, done = reader.read_chunk()
            if not frames or done:
                logger.info("Video processing completed.")
                break

            all_outputs = detector.inference_batch(frames)
            for frame, out in zip(frames, all_outputs):
                frame_id += 1
                logger.debug(f"Processing frame {frame_id}")

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

                # Save frame
                cv2.imwrite(os.path.join(save_dir, f"{frame_id}.png"), online_img)

                # Display frame in Streamlit (convert BGR to RGB for display)
                online_img_rgb = cv2.cvtColor(online_img, cv2.COLOR_BGR2RGB)
                video_placeholder.image(online_img_rgb, caption=f"Frame {frame_id}", use_container_width=True)

                # Update log display
                log_text = "".join(log_buffer[-10:])  # Show last 10 log messages
                log_placeholder.text_area("Logs", log_text, height=200)

        st.success("Processing complete!")

# Instructions
st.write("Enter a video path and click 'Start Processing' to begin tracking.")