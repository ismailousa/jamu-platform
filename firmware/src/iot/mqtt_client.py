import paho.mqtt.client as mqtt
from typing import Callable

class MQTTClient:
    def __init__(self, broker: str, port: int, topic_commands: str, topic_responses: str):
        """
        Initialize MQTT client.
        :param broker: MQTT broker IP
        :param port: MQTT broker port
        :param topic_commands: Topic to subscribe to for commands
        :param topic_responses: Topic to publish responses to
        """
        self.broker = broker
        self.port = port
        self.topic_commands = topic_commands
        self.topic_responses = topic_responses
        self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, rc):
        """
        Callback when the client connects to the broker.
        """
        print("Connected to MQTT broker")
        self.client.subscribe(self.topic_commands)

    def _on_message(self, client, userdata, msg):
        """
        Callback when a message is received.
        """
        command = msg.payload.decode()
        print(f"Received command: {command}")
        self.client.publish(self.topic_responses, f"Received: {command}")

    def start(self):
        """
        Start the MQTT client.
        """
        self.client.connect(self.broker, self.port, 60)
        self.client.loop_start()