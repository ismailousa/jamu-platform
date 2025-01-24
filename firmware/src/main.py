import asyncio
import websockets
import json
from camera.camera_server import CameraServer
from motor.motor_control import MotorControl
from led.led_control import LEDControl
import base64
import threading

class SmartCamera:
    def __init__(self):
        self.camera = CameraServer()
        # self.motor = MotorControl(pin=18)
        self.led = LEDControl()

        # Start a thread to check for camera changes
        self.camera_thread = threading.Thread(target=self.camera.check_camera_change)
        self.camera_thread.daemon = True
        self.camera_thread.start()

    async def handle_client(self, websocket):
        """
        Handle WebSocket client connections.
        """
        print(f"New client connected: {websocket.remote_address}")  # Debug message
        async for message in websocket:
            try:
                print(f"Received command: {message}")  # Debug message
                command = message.strip().lower()
                response = await self.process_command(command, websocket)
                await websocket.send(json.dumps(response))
            except Exception as e:
                print(f"Error processing command: {e}")  # Debug message
                await websocket.send(json.dumps({"status": "error", "message": str(e)}))


    async def process_command(self, command, websocket):
        """
        Process a command and return a response.
        """
        if command == "/stream start":
            self.camera.start_streaming()
            return {"status": "success", "message": "Streaming started"}

        elif command == "/stream stop":
            self.camera.stop_streaming()
            return {"status": "success", "message": "Streaming stopped"}

        elif command == "/stream frame":
            frame_data = self.camera.get_stream_frame()
            if frame_data:
                frame_base64 = base64.b64encode(frame_data).decode("utf-8")
                return {"status": "success", "frame": frame_base64}
            else:
                return {"status": "error", "message": "No frame available"}

        elif command == "/capture picture":
            filename = self.camera.capture_picture()
            with open(filename, "rb") as f:
                image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            return {"status": "success", "image": image_base64}

        elif command == "/record start":
            self.camera.start_recording()
            return {"status": "success", "message": "Recording started"}

        elif command == "/record stop":
            self.camera.stop_recording()
            return {"status": "success", "message": "Recording stopped"}

        elif command == "/led on":
            self.led.toggle("ON")
            return {"status": "success", "message": "LED turned on"}

        elif command == "/led off":
            self.led.toggle("OFF")
            return {"status": "success", "message": "LED turned off"}

        # elif command.startswith("/motor set"):
        #     try:
        #         angle = int(command.split()[2])
        #         self.motor.set_angle(angle)
        #         return {"status": "success", "message": f"Motor set to {angle} degrees"}
        #     except (IndexError, ValueError):
        #         return {"status": "error", "message": "Invalid motor angle"}

        elif command == "/status":
            return {
                "status": "success",
                "streaming": self.camera.streaming,
                "recording": self.camera.recording,
                "led": "on" if self.led.is_on() else "off",
                "motor_angle": 'NOTSET', #self.motor.get_angle(),
                "camera_type": self.camera.camera_type,
            }

        elif command == "/help":
            return {
                "status": "success",
                "commands": [
                    "/stream start",
                    "/stream stop",
                    "/stream frame",
                    "/capture picture",
                    "/record start",
                    "/record stop",
                    "/led on",
                    "/led off",
                    "/motor set <angle>",
                    "/status",
                    "/help",
                ],
            }

        else:
            return {"status": "error", "message": "Invalid command"}

async def start_websocket_server():
    """
    Start the WebSocket server.
    """
    camera = SmartCamera()

    # Create a wrapper function to bind the instance
    async def handler(websocket):
        await camera.handle_client(websocket)

    # Start the WebSocket server with the wrapper function
    async with websockets.serve(handler, "0.0.0.0", 8765):
        print("WebSocket server started on ws://0.0.0.0:8765")
        await asyncio.Future()  

if __name__ == "__main__":
    asyncio.run(start_websocket_server())
