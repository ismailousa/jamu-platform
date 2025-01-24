import RPi.GPIO as GPIO

class MotorControl:
    def __init__(self, pin: int):
        """
        Initialize motor control.
        :param pin: GPIO pin connected to the motor
        """
        self.pin = pin
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.OUT)
        self.pwm = GPIO.PWM(self.pin, 50)  # 50 Hz for servo
        self.pwm.start(0)

    def set_angle(self, angle: int):
        """
        Set motor angle (0-180 degrees).
        :param angle: Desired angle
        """
        if angle < 0 or angle > 180:
            raise ValueError("Angle must be between 0 and 180 degrees.")

        duty_cycle = (angle / 18) + 2  # Convert angle to duty cycle
        self.pwm.ChangeDutyCycle(duty_cycle)
        print(f"Motor angle set to {angle} degrees")

    def cleanup(self):
        """
        Clean up GPIO.
        """
        self.pwm.stop()
        GPIO.cleanup()