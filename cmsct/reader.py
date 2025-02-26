import cv2
from loguru import logger
 
class VideoReader:
    """A simple video reader class to stream frames chunk-wise from video files or cameras.
 
    Args:
        video_source (str or int): Path to video file or camera index (e.g., 0 for default camera)
        chunk_size (int, optional): Number of frames per chunk. Defaults to 60.
    """
    def __init__(self, video_source: str, chunk_size: int = 60) -> None:
        """Initialize the video reader with source and chunk size."""
        self.video_source = video_source
        self.chunk_size = chunk_size
        self.frame_number = 0
        self.all_frames_read = False
        self.cap = None
        self._initialize_video()
 
    def _initialize_video(self) -> None:
        """Initialize video capture and set properties."""
        try:
            self.cap = cv2.VideoCapture(self.video_source)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
 
            if not self.cap.isOpened():
                raise Exception(f"Failed to open video source: {self.video_source}")
            logger.info(f"Video source {self.video_source} opened. "
                        f"Resolution: {self.width}x{self.height}, FPS: {self.fps}, "
                        f"Total frames: {self.total_frames}")
 
        except Exception as e:
            logger.error(f"Error initializing video source {self.video_source}: {str(e)}")
            raise
 
    def read_chunk(self) -> tuple[list, bool]:
        """Read a chunk of frames from the video source.
 
        Returns:
            tuple: (list of frames, boolean indicating if all frames are read)
        """
        if self.all_frames_read or self.cap is None:
            return [], True
 
        frames = []
        for _ in range(self.chunk_size):
            ret, frame = self.cap.read()
            if not ret:
                self.all_frames_read = True
                logger.info("All frames read from video source")
                break
            frames.append(frame)
            self.frame_number += 1
 
        if self.all_frames_read or (self.total_frames > 0 and self.frame_number >= self.total_frames):
            self.all_frames_read = True
            self.cap.release()
            logger.debug(f"Finished reading video. Total frames processed: {self.frame_number}")
 
        logger.info(f"Read chunk of {len(frames)} frames")
        return frames, self.all_frames_read
 
    def release(self) -> None:
        """Release the video capture resource."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            logger.debug(f"Video source {self.video_source} released")
 
    def __del__(self) -> None:
        """Destructor to ensure resources are released."""
        self.release()
 
# Example usage:
if __name__ == "__main__":
    # For video file
    video_path = "dependencies/ByteTrack/videos/palace.mp4"
    reader = VideoReader(video_path, chunk_size=60)
    while True:
        frames, done = reader.read_chunk()
        if not frames or done:
            break
        #print(f"Processed chunk with {len(frames)} frames")
    # For camera (using camera index 0)
    # reader = VideoReader(0, chunk_size=30)
    # while True:
    #     frames, done = reader.read_chunk()
    #     if not frames or done:
    #         break
    #     print(f"Processed chunk with {len(frames)} frames")