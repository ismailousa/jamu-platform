#!/bin/bash

# Exit immediately if any command fails
set -e

# Check if the script is run as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run this script as root or with sudo."
  exit 1
fi

# Update and install system dependencies
echo "Updating system packages..."
sudo apt update

echo "Installing system dependencies..."
sudo apt install -y python3-pip libcamera-apps ffmpeg  # Added ffmpeg for USB camera support

# Optionally create a Python virtual environment
read -p "Do you want to create a Python virtual environment? (y/n): " create_venv
if [[ "$create_venv" =~ ^[Yy]$ ]]; then
  echo "Creating Python virtual environment..."
  python3 -m venv venv
  source venv/bin/activate
  echo "Virtual environment activated."
fi

# Install Python libraries
echo "Installing Python dependencies..."
pip3 install paho-mqtt websockets RPi.GPIO opencv-python numpy  # Added opencv-python and numpy

# Check if the installation was successful
if [ $? -eq 0 ]; then
  echo "Firmware installation complete!"
else
  echo "Firmware installation failed. Please check the logs for errors."
  exit 1
fi