# Raspberry Pi + Laptop Setup Guide

## 1. Network Topology
- Laptop: runs natural-language pipeline (`main.py`) and sends command JSON to Pi.
- Raspberry Pi: receives command JSON (`receive_command.py`) and forwards to Pixhawk via MAVLink.
- Camera stream: Pi camera -> RTSP/UDP -> laptop (`--stream-url`).

## 2. Raspberry Pi Setup
1. Connect Pi to laptop hotspot and check Pi IP.
```bash
hostname -I
```
2. Install runtime dependencies on Pi.
```bash
sudo apt update
sudo apt install -y python3-pip python3-opencv
pip3 install pymavlink
```
3. If using USB Pixhawk serial, verify device path.
```bash
ls /dev/ttyACM* /dev/ttyUSB*
```
4. Update `config/parameters.json` on Pi:
- `drone.connection_string`: e.g. `/dev/ttyACM0` or `udp:127.0.0.1:14550`
- `communication.listen_host`: `0.0.0.0`
- `communication.port`: e.g. `5005`
5. Start command receiver on Pi.
```bash
python3 communication/receive_command.py --host 0.0.0.0 --port 5005
```

## 3. Camera Streaming From Pi
- Example with `libcamera-vid` (RTSP via `rtsp-simple-server` or equivalent):
```bash
libcamera-vid -t 0 --inline --width 640 --height 480 --framerate 20 -o tcp://0.0.0.0:8554
```
- If you already have RTSP service, note URL like:
`rtsp://<PI_IP>:8554/cam`

## 4. Laptop Setup
1. Install dependencies:
```bash
pip install opencv-python ultralytics
```
2. Run pipeline with network stream + Pi target:
```bash
python main.py "follow the man in front" \
  --camera \
  --stream-url "rtsp://<PI_IP>:8554/cam" \
  --send \
  --target-host "<PI_IP>" \
  --target-port 5005
```

## 5. Quick Validation Checklist
- Pi receiver terminal shows `Connected by ...`.
- Receiver logs `Velocity command sent ...` during tracking.
- If no `pymavlink`, receiver prints dry-run only.
- If YOLO model is missing, pipeline falls back to HOG detection automatically.

## 6. Safety Order
1. SITL simulation
2. Bench test (props removed)
3. Tethered low-altitude test
4. Open-space flight with observer
