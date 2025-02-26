import streamlit as st
import time
from cmsct.main import process_video  # Import the process_video function
import os 

def main():
    st.title("CMSCT APSIS Demo App Video Viewer")

    # Input for video path
    video_path = st.text_input(
        "Enter the path to the video file:", 
        "dependencies/ByteTrack/videos/palace.mp4"
    )
    
    # Create a single placeholder for displaying frames
    frame_placeholder = st.empty()
    
    # Process button
    process_button = st.button("Process Video")
    
    if video_path and process_button:
        if os.path.exists(video_path):
            with st.spinner("Processing video..."):
                # Get frames from the generator
                frame_generator = process_video(video_path)
                
                # Counter for frames
                frame_count = 0
                
                # Display frames one at a time in the same placeholder
                for frame in frame_generator:
                    frame_count += 1
                    # Update the same placeholder with the new frame
                    frame_placeholder.image(
                        frame,
                        channels="BGR",
                        use_container_width=True,
                        caption=f"Frame {frame_count}"
                    )
                    time.sleep(0.1)  # Adjust speed of frame updates
                    
                if frame_count > 0:
                    st.success(f"Video processed successfully. Displayed {frame_count} frames.")
                else:
                    st.warning("No frames were processed from the video.")
        else:
            st.error("Video file not found. Please check the path.")

if __name__ == "__main__":
    main()