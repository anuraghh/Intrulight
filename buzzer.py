import pigpio
import time

PI_IP = '192.168.137.78'
PIR_PIN = 17
BUZ_PIN = 27

# Connect to Raspberry Pi
print(f"Connecting to Raspberry Pi at {PI_IP}...")
pi = pigpio.pi(PI_IP)

if not pi.connected:
    print("Connection Failed. Check IP address and if 'sudo pigpiod' is running on the Pi.")
    exit()

print("Connection Successful")

# Setup pins
pi.set_mode(PIR_PIN, pigpio.INPUT)
pi.set_mode(BUZ_PIN, pigpio.OUTPUT)
pi.set_pull_up_down(PIR_PIN, pigpio.PUD_DOWN)

print("Waiting for motion...")

try:
    while True:
        motion_detected = pi.read(PIR_PIN)
        
        if motion_detected:
            print("Motion detected! Buzzing...")
            # Turn on buzzer
            pi.write(BUZ_PIN, 1)
            time.sleep(2)  # Buzzer ON for 2 seconds
            pi.write(BUZ_PIN, 0)
        else:
            # Make sure buzzer is off if no motion
            pi.write(BUZ_PIN, 0)
        
        time.sleep(2)  # Poll PIR frequently

except KeyboardInterrupt:
    print("\nStopping...")
    pi.write(BUZ_PIN, 0)
    pi.stop()
