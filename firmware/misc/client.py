import asyncio
import websockets
import json

uri = "ws://192.168.178.65:8765" 

async def interactive_websocket_client():
    # uri = "ws://localhost:8765" 
    async with websockets.connect(uri) as websocket:
        print("Connected to WebSocket server. Type commands to interact with the server.")
        print("Type 'exit' to quit.")

        while True:
            # Get user input
            command = input("Enter command: ").strip()

            # Exit the loop if the user types 'exit'
            if command.lower() == "exit":
                print("Exiting...")
                break

            # Send the command to the server
            await websocket.send(command)
            print(f"Sent command: {command}")

            # Wait for the response
            response = await websocket.recv()
            print(f"Received response: {response}")

            # Optionally, parse the JSON response
            try:
                response_data = json.loads(response)
                print("Parsed response:", response_data)
            except json.JSONDecodeError:
                print("Response is not in JSON format.")

# Run the interactive client
if __name__ == "__main__":
    asyncio.run(interactive_websocket_client())