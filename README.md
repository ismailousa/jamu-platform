# jamu-platform
A comprehensive platform that streamlines the intake of eye care patients and pre-screening of ocular conditions


## Camera Firmware

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ismailousa/iot-camera-firmware.git
   cd iot-camera-firmware

2. Run the installation script:
   ```bash
    chmod +x install_firmware.sh
    ./install_firmware.sh

3. Run the firmware:

 ```bash
    iot-camera-firmware
    Copy

---

### **How It Works**
1. **System-Level Installation**:
   - The `install_firmware.sh` script updates the system packages and installs `python3-pip` and `libcamera-apps`.

2. **Python Dependency Installation**:
   - The script runs `pip install .` to install the Python dependencies listed in `pyproject.toml`.

3. **Run the Firmware**:
   - After installation, users can run the firmware using the `iot-camera-firmware` command (defined in `pyproject.toml`).

---