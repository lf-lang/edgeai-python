import cv2
import threading
import time

class VideoCapture:
    """
    A class to asynchronously capture video frames using OpenCV and threading.
    Maintains only the most recent frame in memory.
    """

    def __init__(self, source):
        """
        Initialize the video capture object.
        Args:
            source: The video source (file path or camera index).
        """
        start = time.time()
        self.cap = cv2.VideoCapture(source, cv2.CAP_ANY)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source: {source}")
        
        # Initialize frame buffer and synchronization
        self._current_frame = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
        self._is_released = False
        
        # Cache FPS value during initialization
        self._fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Set buffer size to improve performance
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Start the frame reader thread
        self.thread = threading.Thread(target=self._frame_reader, daemon=True)
        self.thread.start()
        print(f"VideoCapture initialized in {time.time() - start:.2f} seconds")

    def _frame_reader(self):
        """
        Continuously reads frames from the video source in a separate thread,
        updating the most recent frame at natural FPS pacing.
        """
        frame_interval = (1.0 / self._fps) if (self._fps and self._fps > 0) else 0.033
        while not self._stop_event.is_set():
            try:
                with self._lock:
                    if self._is_released or not self.cap.isOpened():
                        break
                    t_start = time.time()
                    ret, frame = self.cap.read()
                    
                    if not ret:
                        if self._stop_event.is_set() or self._is_released:
                            break
                        # Video file reached EOF: rewind to start (loop)
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self.cap.read()
                        if not ret:
                            print("Warning: Failed to read frame after rewind. Stopping thread.")
                            break

                    self._current_frame = frame

                # Pacing: use _stop_event.wait so shutdown wakes up immediately
                elapsed = time.time() - t_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    if self._stop_event.wait(timeout=sleep_time):
                        break
            except Exception:
                # Catch any unexpected error during interpreter shutdown
                break

    def read(self):
        """
        Retrieve the latest frame.
        Returns:
            The most recent frame or None if no frame is available.
        """
        with self._lock:
            return self._current_frame

    def is_opened(self):
        """
        Check if the video capture is still open.
        Returns:
            True if the video capture is open, False otherwise.
        """
        with self._lock:
            if self._is_released or self.cap is None:
                return False
            try:
                return self.cap.isOpened()
            except Exception:
                return False

    def release(self):
        """
        Release the video capture and stop the frame reading thread safely and idempotently.
        """
        self._stop_event.set()  # Signal the thread to stop immediately

        # Wait for thread with a short timeout to prevent deadlocks on shutdown
        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=0.3)
        
        with self._lock:
            if not self._is_released:
                self._is_released = True
                try:
                    if self.cap is not None and self.cap.isOpened():
                        self.cap.release()
                except Exception:
                    pass
                self._current_frame = None

    def get_fps(self):
        """
        Get the frames per second (FPS) of the video source.
        Returns:
            The cached FPS value as a float.
        """
        return self._fps