import socket
import cv2
import numpy as np

# Port for the server
port = 5001

def send_command(server_ip, command):
    """
    Send a command to the Raspberry Pi server.
    :param server_ip: IP address of the Raspberry Pi
    :param command: Command to send (e.g., "/led ON", "/stream start", "/stream stop")
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((server_ip, port))
        client_socket.sendall(command.encode())
        response = client_socket.recv(1024).decode()
        print(response)

def start_stream(server_ip):
    """
    Start receiving and displaying the video stream.
    :param server_ip: IP address of the Raspberry Pi
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((server_ip, port))
        client_socket.sendall(b"/stream start")
        response = client_socket.recv(1024).decode()
        print(response)

        while True:
            # Receive the frame size
            frame_size_data = client_socket.recv(4)
            if not frame_size_data:
                break

            frame_size = int.from_bytes(frame_size_data, byteorder='big')

            # Receive the frame data
            frame_data = b""
            while len(frame_data) < frame_size:
                packet = client_socket.recv(frame_size - len(frame_data))
                if not packet:
                    break
                frame_data += packet

            # Decode the frame
            frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                cv2.imshow("Video Stream", frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

def print_help():
    """
    Print available commands.
    """
    print("Available commands:")
    print("  /led <ON/OFF>  - Turn the LED on or off")
    print("  /stream start  - Start video streaming")
    print("  /stream stop   - Stop video streaming")
    print("  /exit          - Exit the client")

def main():
    SERVER_IP = "192.168.178.68"  # Replace with your Raspberry Pi's IP address

    print("Client started. Type /help for a list of commands.")
    while True:
        command = input("> ").strip().lower()
        if command.startswith("led"):
            try:
                _, state = command.split()
                if state.upper() in ["ON", "OFF"]:
                    send_command(SERVER_IP, f"/led {state.upper()}")
                else:
                    print("Invalid LED state. Use 'ON' or 'OFF'.")
            except ValueError:
                print("Invalid command. Usage: led <ON/OFF>")
        elif command == "stream start":
            start_stream(SERVER_IP)
        elif command == "stream stop":
            send_command(SERVER_IP, "/stream stop")
        elif command == "help":
            print_help()
        elif command == "exit":
            print("Exiting client.")
            break
        else:
            print("Invalid command. Type /help for a list of commands.")

if __name__ == "__main__":
    main()