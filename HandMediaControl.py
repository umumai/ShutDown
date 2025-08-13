import cv2
import mediapipe as mp
import subprocess
import time


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Camera init with fallback
capture = cv2.VideoCapture(0)
if not capture.isOpened():
    capture = cv2.VideoCapture(1)

# MediaPipe Hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)


def toggle_music_play_pause():
    """Toggle play/pause in Spotify (macOS) via AppleScript."""
    try:
        subprocess.run(["osascript", "-e", "tell application \"Spotify\" to playpause"], check=True)
        return True
    except subprocess.CalledProcessError:
        return False


# motion-ish tracking state
prev_center_x = None        # previous hand center x in pixels
move_start_time = None      # when rightward motion started
last_trigger_time = 0       # last time an action triggered

# Tunables
PER_FRAME_DELTA_PX = 6      # minimal rightward px/frame to count as motion
SUSTAIN_SECONDS = 2.0       # must maintain rightward motion this long to trigger
COOLDOWN_SECONDS = 3.0      # cooldown after a trigger


def compute_center_x(hand_landmarks, frame_width):
    xs = [lm.x * frame_width for lm in hand_landmarks.landmark]
    return sum(xs) / len(xs)


def draw_status_bar(img, text, color):
    """Draw a filled status bar at the bottom with the provided text and color."""
    h, w, _ = img.shape
    bar_h = 40
    cv2.rectangle(img, (0, h - bar_h), (w, h), color, -1)
    cv2.putText(img, text, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


while True:
    ret, frame = capture.read()
    if not ret:
        print("Failed to capture image")
        break

    # Optional mirror for user-friendly view
    frame = cv2.flip(frame, 1)

    # Process with MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    status_text = ""
    hand_bar_text = "No hand detected"
    hand_bar_color = (0, 0, 255)  # red by default

    if results.multi_hand_landmarks:
        h, w, _ = frame.shape
        # Use the first hand for motion control
        hand_landmarks = results.multi_hand_landmarks[0]

        # Handedness label and confidence (if available)
        if results.multi_handedness:
            handed = results.multi_handedness[0].classification[0]
            hand_bar_text = f"{handed.label} Hand | conf: {handed.score:.2f}"
            hand_bar_color = (0, 170, 0)  # green
        else:
            hand_bar_text = "Hand detected"
            hand_bar_color = (0, 170, 0)

        # Draw landmarks
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        # Compute center x and detect rightward motion
        center_x = compute_center_x(hand_landmarks, w)
        now = time.time()

        if prev_center_x is None:
            prev_center_x = center_x

        delta_x = center_x - prev_center_x
        prev_center_x = center_x

        # Rightward motion detected
        if delta_x > PER_FRAME_DELTA_PX:
            if move_start_time is None:
                move_start_time = now
            elapsed = now - move_start_time
            status_text = f"Moving right: {elapsed:.1f}s"
            # Trigger if sustained and not in cooldown
            if elapsed >= SUSTAIN_SECONDS and (now - last_trigger_time) >= COOLDOWN_SECONDS:
                ok = toggle_music_play_pause()
                last_trigger_time = now
                move_start_time = None
                status_text = "Play/Pause triggered" if ok else "Trigger failed"
        else:
            # Reset timer on strong left motion, or slowly when movement pauses
            if delta_x < -PER_FRAME_DELTA_PX:
                move_start_time = None
                status_text = "Moving left: reset"

    # Draw status bar (hand presence)
    draw_status_bar(frame, hand_bar_text, hand_bar_color)
    # Overlay motion status (if any)
    if status_text:
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow('Hand Media Control', frame)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC to exit
        break

# Cleanup
capture.release()
# cv2.destroyAllWindows()