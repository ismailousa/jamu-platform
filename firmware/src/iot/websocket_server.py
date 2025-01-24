import asyncio
import websockets
from typing import Callable

class WebSocketServer:
    def __init__(self, host: str, port: int):
        """
        Initialize WebSocket server.
        :param host: Host IP to bind to
        :param port: Port to listen on
        """
        self.host = host
        self.port = port

    async def _handle_client(self, websocket, path):
        """
        Handle WebSocket client connections.
        """
        async for message in websocket:
            print(f"Received command: {message}")
            await websocket.send(f"Received: {message}")

    def start(self):
        """
        Start the WebSocket server.
        """
        print(f"WebSocket server started on {self.host}:{self.port}")
        asyncio.get_event_loop().run_until_complete(
            websockets.serve(self._handle_client, self.host, self.port)
        )
        asyncio.get_event_loop().run_forever()