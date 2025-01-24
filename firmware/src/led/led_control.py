import os

LED_PATH = "/sys/class/leds/ACT/brightness"

class LEDControl:
    def __init__(self):
        # Disable the LED trigger to allow manual control
        os.system("echo none | sudo tee /sys/class/leds/ACT/trigger")

    def toggle(self, state: str):
        """
        Toggle the ACT LED on or off.
        :param state: 'ON' or 'OFF'
        """
        if state.upper() not in ["ON", "OFF"]:
            raise ValueError("State must be 'ON' or 'OFF'.")

        try:
            with open(LED_PATH, "w") as led_file:
                led_file.write("1" if state.upper() == "ON" else "0")
            print(f"LED turned {state.upper()}")
        except Exception as e:
            print(f"Error controlling LED: {e}")