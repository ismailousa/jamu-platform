#!/bin/bash

# Configuration variables
P2P_INTERFACE="p2p0"  # Wi-Fi Direct interface
P2P_SSID="RPi-WiFi-Direct"
P2P_PASSPHRASE="12345678"  # Change this to a secure passphrase

echo "Starting Wi-Fi Direct server at startup..."

# Create a separate wpa_supplicant configuration file for Wi-Fi Direct
P2P_CONF="/etc/wpa_supplicant/wpa_supplicant_p2p.conf"

cat > $P2P_CONF <<EOF
ctrl_interface=DIR=/var/run/wpa_supplicant_p2p GROUP=netdev
update_config=1
country=DE

p2p_no_group_iface=1
EOF

# Start a new wpa_supplicant instance for Wi-Fi Direct
sudo wpa_supplicant -B -i $P2P_INTERFACE -c $P2P_CONF -D nl80211

# Configure the P2P group
sudo wpa_cli -i $P2P_INTERFACE p2p_group_add persistent=0
sudo wpa_cli -i $P2P_INTERFACE set p2p_ssid_postfix "$P2P_SSID"
sudo wpa_cli -i $P2P_INTERFACE set p2p_passphrase "$P2P_PASSPHRASE"

# Start the DHCP server for the P2P interface
sudo dhcpd $P2P_INTERFACE

echo "Wi-Fi Direct server started at startup!"
echo "SSID: $P2P_SSID"
echo "Passphrase: $P2P_PASSPHRASE"