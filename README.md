# ShutDown

A gesture-controlled computer sleep system using hand tracking and computer vision.

## About

ShutDown is a Python application that uses MediaPipe and OpenCV to detect specific hand gestures via webcam and trigger computer sleep functionality. The system recognizes two distinct gestures:

- **Middle finger gesture** - Puts computer to sleep
- **Pinkie finger gesture** - Puts computer to sleep

## Features

- **Real-time hand tracking** using MediaPipe
- **Precise gesture recognition** with strict finger positioning requirements
- **Visual feedback** with on-screen status indicators
- **Safety features** including 3-second cooldown between gestures
- **Test mode** for validating gesture detection without triggering sleep
- **Cross-platform sleep commands** (optimized for macOS)

## How It Works

1. Captures live video feed from webcam
2. Processes hand landmarks using MediaPipe
3. Analyzes finger positions to detect specific gestures
4. Triggers system sleep after gesture confirmation
5. Provides visual and console feedback

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- MediaPipe (`mediapipe`)
- Webcam/Camera access

## Usage

```bash
python3 Handtracker.py
```

- Show middle finger or pinkie finger to trigger sleep
- Press 'q' to quit the application
- Status indicators show detection state in real-time

## Technical Details

The system uses MediaPipe's hand landmark detection to identify 21 key points on each hand. Gesture recognition is based on relative positions of finger tips and PIP (Proximal Interphalangeal) joints, ensuring accurate detection while preventing false positives.

---

*Developed for hands-free computer control and accessibility purposes.*
