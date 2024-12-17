import cv2
import queue
import threading

class VideoCapture:
    """
    A class to asynchronously capture video frames using OpenCV and threading.
    Only the most recent frame is kept in the queue.
    """
    def __init__(self, source):
        """
        Initialize the video capture object.
        Args:
            source: The video source (file path or camera index).
        """
        self.cap = cv2.VideoCapture(source, cv2.CAP_ANY)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source: {source}")
        self.q = queue.Queue(maxsize=1)  # Max size of 1 to hold only the latest frame
        self._stop_event = threading.Event()
        self.lock = threading.Lock()  # Lock to synchronize access to the queue
        self.thread = threading.Thread(target=self._frame_reader, daemon=True)
        self.thread.start()

    def _frame_reader(self):
        """
        Continuously reads frames from the video source in a separate thread,
        keeping only the most recent frame.
        """
        while not self._stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                break
            # Synchronize access to the queue
            with self.lock:
                try:
                    self.q.get_nowait()  # Remove the previous frame if it exists
                except queue.Empty:
                    pass
                self.q.put_nowait(frame)  # Add the latest frame

    def read(self):
        """
        Retrieve the latest frame from the queue.
        Returns:
            The most recent frame or None if no frame is available.
        """
        with self.lock:  # Synchronize access to the queue
            try:
                return self.q.get_nowait()
            except queue.Empty:
                return None

    def is_opened(self):
        """
        Check if the video capture is still open.
        Returns:
            True if the video capture is open, False otherwise.
        """
        return self.cap.isOpened()

    def release(self):
        """
        Release the video capture and stop the frame reading thread.
        """
        self._stop_event.set()
        if self.thread.is_alive():
            self.thread.join()
        if self.cap.isOpened():
            self.cap.release()
        # Clear the queue to release memory
        with self.lock:  # Synchronize access to the queue
            while not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break

    def get_fps(self):
        """
        Get the frames per second (FPS) of the video source.
        Returns:
            The FPS value as a float.
        """
        return self.cap.get(cv2.CAP_PROP_FPS)