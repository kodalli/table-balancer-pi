import asyncio
import cv2
from picamera2 import Picamera2
from threading import Thread, Lock
from websockets import serve
import time

class CameraCapture:
    def __init__(self, fps=24, jpeg_quality=60, resolution=(640, 480), postprocess=None):
        """
        Initialize the camera capture with specified FPS, JPEG quality, and resolution.
        """
        self.picam2 = Picamera2()
        self.resolution = resolution
        self.picam2.configure(self.picam2.create_still_configuration(main={"size": self.resolution}))
        self.latest_frame = None
        self.lock = Lock()
        self.running = False
        self.fps = max(1, min(30, fps))  # Ensure FPS is between 1 and 30
        self.jpeg_quality = jpeg_quality  # JPEG quality (1-100)
        self.postprocess = postprocess

    def start(self):
        """Start the camera and the frame capture thread."""
        self.picam2.start()
        self.running = True
        self.capture_thread = Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()
        print(f"[{self._current_time()}] Camera capture started with {self.fps} FPS and JPEG quality {self.jpeg_quality}.")

    def _capture_frames(self):
        """Continuously capture frames at the specified FPS and update the latest_frame."""
        interval = 1.0 / self.fps
        while self.running:
            start_time = time.time()
            frame = self.picam2.capture_array()

            if self.postprocess:
                frame = self.postprocess(frame)

            # Encode the frame as JPEG with the specified quality
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            ret, encoded_frame = cv2.imencode('.jpg', frame, encode_params)
            if ret:
                with self.lock:
                    self.latest_frame = encoded_frame.tobytes()
            else:
                print(f"[{self._current_time()}] Failed to encode frame.")
            elapsed = time.time() - start_time
            time.sleep(max(0, interval - elapsed))

    def stop(self):
        """Stop the camera and the frame capture thread."""
        self.running = False
        self.capture_thread.join()
        self.picam2.stop()
        print(f"[{self._current_time()}] Camera capture stopped.")

    def get_latest_frame(self):
        """Retrieve the latest captured frame in binary format."""
        with self.lock:
            return self.latest_frame

    @staticmethod
    def _current_time():
        """Return the current time formatted for logging."""
        return time.strftime('%Y-%m-%d %H:%M:%S')

def tracker(frame):
    """
    Post-process the frame to track objects (e.g., a ball).
    This function mirrors the logic from your original tracker.py script.
    
    :param frame: Input frame as a NumPy array (BGR).
    :return: Processed frame as a NumPy array (BGR).
    """
    THRESHOLD = 120

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold the image to isolate the ball
    _, thresh = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    # Find contours of the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the enclosing circle or bounding box for the largest object
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)

        if radius > 5:  # Ignore small objects
            # Draw the object on the original frame
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv2.putText(frame, f"Position: ({int(x)}, {int(y)})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return frame

async def stream_frames(websocket):
    """
    Handle incoming WebSocket connections and stream frames to the client.
    """
    client_address = websocket.remote_address
    print(f"[{CameraCapture._current_time()}] Client connected: {client_address}")
    try:
        while True:
            frame = camera_capture.get_latest_frame()
            if frame:
                await websocket.send(frame)  # Send binary data directly
                # Optionally, log frame transmission
                # print(f"[{CameraCapture._current_time()}] Frame sent to {client_address}")
            await asyncio.sleep(1 / camera_capture.fps)  # Control the send rate
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"[{CameraCapture._current_time()}] Error with client {client_address}: {e}")
    finally:
        print(f"[{CameraCapture._current_time()}] Client disconnected: {client_address}")

async def main():
    """
    Initialize the camera capture and start the WebSocket server.
    """
    camera_capture.start()
    print(f"[{CameraCapture._current_time()}] Starting WebSocket server on ws://0.0.0.0:8765")
    async with serve(stream_frames, "0.0.0.0", 8765, max_size=None, max_queue=None):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    # Initialize CameraCapture with desired settings
    camera_capture = CameraCapture(fps=24, jpeg_quality=20, resolution=(640, 480), postprocess=tracker)
    try:
        asyncio.run(main())  # Start the async event loop
    except KeyboardInterrupt:
        print(f"[{CameraCapture._current_time()}] KeyboardInterrupt received. Shutting down...")
    finally:
        camera_capture.stop()

