from pymavlink import mavutil

master = mavutil.mavlink_connection('COM3', baud=115200)

print("Waiting for heartbeat...")
master.wait_heartbeat()
print("Connected to Pixhawk!")

while True:
    msg = master.recv_match(type='HEARTBEAT', blocking=True)
    if msg:
        print("Heartbeat received")
        print(f"Base mode: {msg.base_mode}")
        print(f"Custom mode: {msg.custom_mode}")