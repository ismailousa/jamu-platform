import subprocess
import threading
import time
from typing import Optional
import os

class CameraServer:
    def __init__(self):
        self.streaming: bool = False
        self.recording: bool = False
        self.stream_process: Optional[subprocess.Popen] = None
        self.record_process: Optional[subprocess.Popen] = None
        self.stream_thread: Optional[threading.Thread] = None
        self.client_socket = None
        self.camera_type = self.detect_camera_type()  # Detect camera type on initialization

    def detect_camera_type(self):
        """
        Detect whether a USB or CSI camera is connected.
        """
        try:
            # Check for USB camera
            subprocess.run(["v4l2-ctl", "--list-devices"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return "USB"
        except subprocess.CalledProcessError:
            try:
                # Check for CSI camera
                subprocess.run(["libcamera-hello", "--list-cameras"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return "CSI"
            except subprocess.CalledProcessError:
                return None

    def start_streaming(self):
        """
        Start streaming video using the appropriate tool for the detected camera.
        """
        if self.streaming:
            print("Streaming is already active.")
            return

        if self.camera_type == "USB":
            self.stream_process = subprocess.Popen(
                ["ffmpeg", "-f", "v4l2", "-i", "/dev/video0", "-f", "mpegts", "-"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        elif self.camera_type == "CSI":
            self.stream_process = subprocess.Popen(
                ["libcamera-vid", "-t", "0", "-o", "-"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        else:
            print("No camera detected.")
            return

        self.streaming = True
        print(f"Streaming started with {self.camera_type} camera.")

    def stop_streaming(self):
        """
        Stop the video streaming.
        """
        if not self.streaming:
            print("Streaming is not active.")
            return

        if self.stream_process:
            self.stream_process.terminate()
            self.stream_process = None
        self.streaming = False
        print("Streaming stopped.")

    def capture_picture(self, filename: str = "photo.jpg"):
        """
        Capture a picture using the appropriate tool for the detected camera.
        :param filename: Name of the output file
        """
        if self.camera_type == "USB":
            subprocess.run(
                ["ffmpeg", "-f", "v4l2", "-i", "/dev/video0", "-frames:v", "1", filename],
                check=True,
            )
        elif self.camera_type == "CSI":
            subprocess.run(
                ["libcamera-jpeg", "-o", filename],
                check=True,
            )
        else:
            print("No camera detected.")
            return None

        print(f"Picture captured: {filename}")
        return filename

    def start_recording(self, filename: str = "video.h264"):
        """
        Start recording video using the appropriate tool for the detected camera.
        :param filename: Name of the output file
        """
        if self.recording:
            print("Recording is already active.")
            return

        if self.camera_type == "USB":
            self.record_process = subprocess.Popen(
                ["ffmpeg", "-f", "v4l2", "-i", "/dev/video0", "-t", "0", "-c:v", "libx264", filename],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        elif self.camera_type == "CSI":
            self.record_process = subprocess.Popen(
                ["libcamera-vid", "-t", "0", "-o", filename],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        else:
            print("No camera detected.")
            return

        self.recording = True
        print(f"Recording started with {self.camera_type} camera: {filename}")

    def stop_recording(self):
        """
        Stop the video recording.
        """
        if not self.recording:
            print("Recording is not active.")
            return

        if self.record_process:
            self.record_process.terminate()
            self.record_process = None
        self.recording = False
        print("Recording stopped.")

    def get_stream_frame(self):
        """
        Get a single frame from the video stream.
        """
        if not self.streaming or not self.stream_process:
            return None

        frame_data = self.stream_process.stdout.read(1024)
        return frame_data

    def check_camera_change(self):
        """
        Periodically check for camera changes and restart the stream if necessary.
        """
        while True:
            new_camera_type = self.detect_camera_type()
            if new_camera_type != self.camera_type:
                print(f"Camera changed from {self.camera_type} to {new_camera_type}.")
                self.camera_type = new_camera_type
                if self.streaming:
                    self.stop_streaming()
                    self.start_streaming()
            time.sleep(5)  # Check every 5 seconds