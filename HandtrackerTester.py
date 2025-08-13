#Finger counting status



# Import libraries (Mediapipe and OpenCV)
# Detect each fingers position 

import cv2
import mediapipe as mp
#import subprocess
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

capture = cv2.VideoCapture(0)
if not capture.isOpened():
    capture = cv2.VideoCapture(1)

hands = mp_hands.Hands()

# variables
last_gesture_time = 0 # Control the timing of gestures so it doesn't repeat too quickly
gesture_cooldown = 3  # 3 second cooldown between gestures

def report_finger_status(landmarks):
    # Finger tip and pip landmarks
    # Thumb: 4, 3
    # Index: 8, 6
    # Middle: 12, 10
    # Ring: 16, 14
    # Pinky: 20, 18

    thumb_up = landmarks[4].y < landmarks[3].y
    index_up = landmarks[8].y < landmarks[6].y
    middle_up = landmarks[12].y < landmarks[10].y
    ring_up = landmarks[16].y < landmarks[14].y
    pinky_up = landmarks[20].y < landmarks[18].y

    thumb_down = not thumb_up
    index_down = not index_up
    middle_down = not middle_up
    ring_down = not ring_up
    pinky_down = not pinky_up

    status = []
    if thumb_up:
        status.append("Thumb is up")
    else:
        status.append("Thumb is down")
    if index_up:
        status.append("Index is up")
    else:
        status.append("Index is down")
    if middle_up:
        status.append("Middle is up")
    else:
        status.append("Middle is down")
    if ring_up:
        status.append("Ring is up")
    else:
        status.append("Ring is down")
    if pinky_up:
        status.append("Pinky is up")
    else:
        status.append("Pinky is down")

    if all([thumb_down, index_down, middle_down, ring_down, pinky_down]):
        status.append("All fingers are down")
    if all([thumb_up, index_up, middle_up, ring_up, pinky_up]):
        status.append("High Five!")

        

    print("; ".join(status))

while True:
    # image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    ret, frame = capture.read()
    if not ret:
        print("Failed to capture image")
        break

    frame = cv2.flip(frame, 2) # Flip horizontally = 1 camera flip = 2
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            report_finger_status(hand_landmarks.landmark)

    cv2.imshow('Hand Tracker', frame)

    if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
        break

# clean up lines (prevent from crashing and so on)
capture.release()
cv2.destroyAllWindows()
