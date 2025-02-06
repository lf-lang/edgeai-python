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
        updating the most recent frame.
        """
        while not self._stop_event.is_set():
            if not self.cap.isOpened():
                print("Warning: VideoCapture is closed.")
                break

            ret, frame = self.cap.read()
            
            if not ret:
                print("Warning: Failed to read frame. Stopping thread.")
                break
            
            with self._lock:
                self._current_frame = frame

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
            return self.cap.isOpened()

    def release(self):
        """
        Release the video capture and stop the frame reading thread.
        """
        self._stop_event.set()  # Signal the thread to stop

        if self.thread.is_alive():
            self.thread.join()  # Wait for the thread to exit
        
        with self._lock:
            if self.cap.isOpened():
                self.cap.release()
            
        # Clear the frame buffer
        self._current_frame = None

    def get_fps(self):
        """
        Get the frames per second (FPS) of the video source.
        Returns:
            The cached FPS value as a float.
        """
        return self._fps