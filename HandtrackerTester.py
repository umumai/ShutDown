# Import libraries (Mediapipe and OpenCV)
# Detect each fingers position 

import cv2
import mediapipe as mp
import subprocess
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

# Motion tracking state (screen-right movement to trigger play/pause)
motion_prev_x = None           # previous hand center x (in pixels)
motion_start_time = None       # time when rightward motion started
last_action_time = 0           # last time we triggered an action
move_threshold_px = 6          # minimum per-frame delta x (pixels) to consider as rightward motion
sustain_seconds = 2.0          # must keep moving right for this long to trigger
motion_cooldown = 3.0          # cooldown between triggers (seconds)


def toggle_music_play_pause():
    """Toggle play/pause in Apple Music (macOS) via AppleScript."""
    try:
        subprocess.run(["osascript", "-e", "tell application \"Music\" to playpause"], check=True)
        return True
    except subprocess.CalledProcessError:
        return False

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

    frame = cv2.flip(frame, 2) # Flip horizontally = 1 (any >0). Using 2 also flips horizontally.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        h, w, _ = frame.shape
        # Use the first detected hand for motion (extend to loop if you want per-hand actions)
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            report_finger_status(hand_landmarks.landmark)

            # Compute hand center x (average of all landmark x in pixels)
            xs = [lm.x * w for lm in hand_landmarks.landmark]
            center_x = sum(xs) / len(xs)

            # Rightward motion detection with sustain
            now = time.time()
            if motion_prev_x is None:
                motion_prev_x = center_x
            delta_x = center_x - motion_prev_x
            motion_prev_x = center_x

            status_text = ""
            # Significant right movement
            if delta_x > move_threshold_px:
                if motion_start_time is None:
                    motion_start_time = now
                elapsed = now - motion_start_time
                status_text = f"Moving right: {elapsed:.1f}s"
                # Trigger if sustained and not in cooldown
                if elapsed >= sustain_seconds and (now - last_action_time) > motion_cooldown:
                    ok = toggle_music_play_pause()
                    last_action_time = now
                    motion_start_time = None  # reset to avoid immediate retrigger
                    status_text = "Play/Pause triggered" if ok else "Trigger failed"
            else:
                # Significant left movement cancels, or no strong movement keeps timer running briefly
                if delta_x < -move_threshold_px:
                    motion_start_time = None
                    status_text = "Moving left: reset"

            # Draw status near the top-left
            if status_text:
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Only use the first hand for motion detection to reduce noise
            break

    cv2.imshow('Hand Tracker', frame)

    if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
        break

# clean up lines (prevent from crashing and so on)
capture.release()
cv2.destroyAllWindows()
